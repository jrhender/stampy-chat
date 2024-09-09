from typing import Any, Callable, Dict, List

from langchain.chains import LLMChain, OpenAIModerationChain
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory
from langchain.prompts import (
    BaseChatPromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.pydantic_v1 import Extra
from langchain.schema import AIMessage, BaseMessage, HumanMessage, PromptValue, SystemMessage

from stampy_chat.env import OPENAI_API_KEY, ANTHROPIC_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, SUMMARY_MODEL
from stampy_chat.settings import Settings, MODELS, OPENAI, ANTRHROPIC
from stampy_chat.callbacks import StampyCallbackHandler, BroadcastCallbackHandler, LoggerCallbackHandler
from stampy_chat.followups import StampyChain
from stampy_chat.citations import make_example_selector

from langsmith import Client

if LANGCHAIN_TRACING_V2 == "true":
    if not LANGCHAIN_API_KEY:
        raise Exception("Langsmith tracing is enabled but no api key was provided. Please set LANGCHAIN_API_KEY in the .env file.")
    client = Client()

class ModerationError(ValueError):
    pass

class MessageBufferPromptTemplate(FewShotChatMessagePromptTemplate):
    get_num_tokens: Callable[[str], int]
    max_tokens: int

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        all_messages = super().format_messages(**kwargs)

        messages = []
        remaining_tokens = self.max_tokens
        for message in all_messages:
            tokens = self.get_num_tokens(message.content)
            if tokens > remaining_tokens:
                break

            remaining_tokens -= tokens
            messages.append(message)
        return messages

def ChatMessage(m):
    if m['role'] == 'assistant':
        return AIMessage(**m)
    return HumanMessage(**m)

class PrefixedPrompt(BaseChatPromptTemplate):
    transformer: Callable[[Any], BaseMessage] = lambda i: i
    messages_field: str
    prompt: str

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        history = kwargs[self.messages_field]
        if history and self.prompt:
            return [HumanMessage(content=self.prompt)] + [self.transformer(i) for i in history]
        return []

class LimitedConversationSummaryBufferMemory(ConversationSummaryBufferMemory):
    callbacks: List[StampyCallbackHandler] = []
    max_history: int = 10

    def set_messages(self, history: List[dict]) -> None:
        for callback in self.callbacks:
            callback.on_memory_set_start(history)

        messages = [ChatMessage(m) for m in history if m.get('role') != 'deleted']
        if len(messages) > self.max_history :
            offset = -self.max_history + 1

            pruned = messages[:offset]
            summary = AIMessage(role='assistant', content=self.predict_new_summary(pruned, ''))

            messages = [summary] + messages[offset:]

        self.clear()
        self.chat_memory = ChatMessageHistory(messages=messages)
        self.prune()

        for callback in self.callbacks:
            callback.on_memory_set_end(self.chat_memory)

    def prune(self) -> None:
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while buffer and curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        pass

class ModeratedChatPrompt(ChatPromptTemplate):
    moderation_chain: OpenAIModerationChain = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.moderation_chain:
            self.moderation_chain = OpenAIModerationChain(error=True, openai_api_key=OPENAI_API_KEY)

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        prompt = super().format_prompt(**kwargs)
        try:
            self.moderation_chain.run(prompt.to_string())
        except ValueError as e:
            raise ModerationError(e)
        return prompt

class ChatAnthropicWrapper(ChatAnthropic):
    def _format_params(self, *args, **kwargs):
        first = kwargs['messages'][0]
        if isinstance(first, AIMessage):
            first = SystemMessage(content=first.content)

        messages = [first]
        for m in kwargs['messages'][1:]:
            if m.type != messages[-1].type:
                messages.append(m)
            else:
                messages[-1].content += '\n\n' + m.content
        kwargs['messages'] = messages
        return super()._format_params(*args, **kwargs)

def get_model(**kwargs):
    model = MODELS.get(kwargs.get('model'))
    if not model:
        raise ValueError("No model provided")
    if model.publisher == ANTRHROPIC:
        return ChatAnthropicWrapper(anthropic_api_key=ANTHROPIC_API_KEY, **kwargs)
    if model.publisher == OPENAI:
        return ChatOpenAI(openai_api_key=OPENAI_API_KEY, **kwargs)
    raise ValueError(f'Unsupported model: {kwargs.get("model")}')

class LLMInputsChain(LLMChain):
    inputs: Dict[str, Any] = {}

    def _call(self, inputs: Dict[str, Any], run_manager=None):
        self.inputs = inputs
        return super()._call(inputs, run_manager)

    def _acall(self, inputs: Dict[str, Any], run_manager=None):
        self.inputs = inputs
        return super()._acall(inputs, run_manager)

    def create_outputs(self, llm_result) -> List[Dict[str, Any]]:
        result = super().create_outputs(llm_result)
        return [dict(self.inputs, **r) for r in result]

def make_history_summary(settings):
    model = get_model(
        streaming=False,
        max_tokens=settings.maxHistorySummaryTokens,
        model=settings.completions
    )
    summary_prompt = PrefixedPrompt(
        input_variables=['history'],
        messages_field='history',
        prompt=settings.history_summary_prompt,
        transformer=ChatMessage,
    )
    return LLMInputsChain(
        llm=model,
        verbose=False,
        output_key='history_summary',
        prompt=ModeratedChatPrompt.from_messages([
            summary_prompt,
            ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template(template='Q: {query}', role='user'),
            ]),
            HumanMessage(content="Reply in one sentence only"),
        ]),
    )

def make_prompt(settings, chat_model, callbacks):
    context_template = "\n\n[{{reference}}] {{title}} {{authors | join(', ')}} - {{date_published}} {{text}}\n\n"
    context_prompt = MessageBufferPromptTemplate(
        example_selector=make_example_selector(k=settings.topKBlocks, callbacks=callbacks),
        example_prompt=ChatPromptTemplate.from_template(context_template, template_format="jinja2"),
        get_num_tokens=chat_model.get_num_tokens,
        max_tokens=settings.context_tokens,
        input_variables=['query', 'history'],
    )

    history_prompt = PrefixedPrompt(input_variables=['history'], messages_field='history', prompt=settings.history_prompt)

    query_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=settings.question_prompt),
            HumanMessagePromptTemplate.from_template(
                template='{history_summary}{delimiter}{query}',
                partial_variables={"delimiter": lambda **kwargs: ": " if kwargs.get("history_summary") else ""}
            ),
        ]
    )

    return ModeratedChatPrompt.from_messages([
        SystemMessage(content=settings.context_prompt),
        context_prompt,
        history_prompt,
        query_prompt,
    ])

def make_memory(settings, history, callbacks):
    memory = LimitedConversationSummaryBufferMemory(
        llm=get_model(model=SUMMARY_MODEL),
        max_token_limit=settings.history_tokens,
        max_history=settings.maxHistory,
        chat_memory=ChatMessageHistory(),
        return_messages=True,
        callbacks=callbacks
    )
    memory.set_messages([i for i in history if i.get('role') != 'deleted'])
    return memory

def merge_history(history):
    if not history:
        return history

    messages = []
    current_message = history[0]
    for message in history[1:]:
        if message.get('role') in ['deleted', 'error']:
            continue
        if message.get('role') != current_message.get('role'):
            messages.append(current_message)
            current_message = message
        else:
            current_message['content'] += '\n' + message.get('content', '')
    messages.append(current_message)
    return messages

def run_query(session_id: str, query: str, history: List[Dict], settings: Settings, callback: Callable[[Any], None] = None, followups=True) -> Dict[str, str]:
    callbacks = [LoggerCallbackHandler(session_id=session_id, query=query, history=history)]
    if callback:
        callbacks += [BroadcastCallbackHandler(callback)]

    history = merge_history(history)
    chat_model = get_model(
        streaming=True,
        callbacks=callbacks,
        max_tokens=settings.max_response_tokens,
        model=settings.completions
    )

    history_summary_chain = make_history_summary(settings)
    
    if history:
        history_summary_result = history_summary_chain.invoke({"query": query, 'history': history})
        history_summary = history_summary_result.get('history_summary', '')
    else:
        history_summary = ''

    delimiter = ": " if history_summary else ""

    llm_chain = LLMChain(
        llm=chat_model,
        verbose=False,
        prompt=make_prompt(settings, chat_model, callbacks),
        memory=make_memory(settings, history, callbacks)
    )
    
    chain = history_summary_chain | llm_chain
    if followups:
        chain = chain | StampyChain(callbacks=callbacks)
    
    chain_input = {
        "query": query,
        'history': history,
        'history_summary': history_summary,
        'delimiter': delimiter,
    }
    
    result = chain.invoke(chain_input)

    if callback:
        callback({'state': 'done'})
        callback(None)
    return result

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"