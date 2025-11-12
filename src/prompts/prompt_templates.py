"""
Prompt Templates for ARE-VQA Pipeline
Contains all prompts used by different modules
"""

# =============================================================================
# Module 1: Triage Router Prompts
# =============================================================================

TRIAGE_PROMPT = """You are a Triage Router for a Visual Question Answering system. Your task is to classify the given question along two dimensions:

1. **Complexity**: 
   - "ATOMIC": The question can be answered in a single step
   - "COMPOSITIONAL": The question requires multiple reasoning steps

2. **Knowledge**:
   - "VISUAL": The answer can be determined from visual information in the image alone
   - "KNOWLEDGE-BASED": The answer requires external world knowledge beyond what's visible

Respond ONLY with valid JSON in this exact format:
{{"complexity": "ATOMIC or COMPOSITIONAL", "knowledge": "VISUAL or KNOWLEDGE-BASED"}}

Examples:

Question: "What color is the car?"
{{"complexity": "ATOMIC", "knowledge": "VISUAL"}}

Question: "In which city is this famous landmark located?"
{{"complexity": "ATOMIC", "knowledge": "KNOWLEDGE-BASED"}}

Question: "What is the brand of the laptop next to the red book?"
{{"complexity": "COMPOSITIONAL", "knowledge": "VISUAL"}}

Question: "Who is the manufacturer of the vehicle parked in front of the building?"
{{"complexity": "COMPOSITIONAL", "knowledge": "KNOWLEDGE-BASED"}}

Now classify this question:

Question: "{question}"
"""

# =============================================================================
# Module 2: Context Builder Prompts
# =============================================================================

VISUAL_CONTEXT_PROMPT = """You are analyzing an image to answer a specific question. Provide a detailed description of the image that focuses on information relevant to the question.

Question: {question}

Describe the image, paying special attention to:
1. Main objects and their attributes (colors, sizes, positions)
2. People and their actions
3. Text visible in the image (signs, labels, etc.)
4. Spatial relationships between objects
5. Any other details relevant to answering the question

Provide a clear, detailed description:"""

KNOWLEDGE_AGENT_PROMPT = """You are a Knowledge Agent. Your task is to provide factual, concise information to help answer a visual question.

Question: {question}

Visual Context: {visual_context}

Based on the question and visual context, provide relevant factual knowledge that would help answer this question. Focus on:
- Historical facts, dates, and events
- Geographic information (locations, cities, countries)
- Common knowledge about objects, brands, people, or concepts
- Cultural or contextual information

Provide 2-3 sentences of relevant factual information. Be concise and factual:"""

QUERY_GENERATION_PROMPT = """You are helping to answer a visual question that requires external knowledge. Based on the question and visual context, generate a focused search query to find the necessary information.

Question: {question}

Visual Context: {visual_context}

Generate a concise search query (2-8 words) that will help find the answer. Output ONLY the search query, nothing else:"""

# =============================================================================
# Module 3: Query Planner Prompts
# =============================================================================

PLANNER_PROMPT = """You are a Query Planner for a Visual Question Answering system. Your task is to break down a complex question into simpler, atomic sub-questions that can be answered sequentially.

Question: {question}

Context:
{context}

Break this question into a numbered list of simple, answerable sub-questions. Each sub-question should:
1. Be answerable with the available visual and contextual information
2. Build upon previous sub-questions
3. Lead logically to the final answer

Format your response as a numbered list. For example:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question]

If the question is already simple enough, just return it as a single item:
1. {question}

Now, break down the question:"""

# =============================================================================
# Module 4: Tool Executor Prompts
# =============================================================================

EXECUTOR_VISUAL_PROMPT = """You are a visual analysis expert. Carefully examine the image and answer the question based on what you observe.

Context (for additional information):
{context}

Question: {question}

Instructions:
- Look carefully at ALL details in the image, including small objects, text, and subtle features
- Pay special attention to people, objects in their hands/mouths, clothing, signs, numbers, brands, etc.
- If you see something relevant to the question, describe it clearly
- Be specific and factual about what you observe
- If you're uncertain, describe what you see rather than saying "nothing" or "not visible"

Answer based on your careful visual observation:"""

EXECUTOR_MC_PROMPT = """Answer the following multiple-choice question based on the image and provided context.

Context:
{context}

Question: {question}

Choices:
{choices}

Select the best answer and respond with ONLY the letter (A, B, C, or D):"""

# =============================================================================
# Module 5: Synthesizer Prompts
# =============================================================================

SYNTHESIZER_PROMPT = """You are synthesizing the final answer to a multiple-choice question based on gathered information and intermediate reasoning steps.

Original Question: {question}

Context:
{context}

Intermediate Reasoning:
{intermediate_answers}

Multiple Choice Options:
{choices}

Based on all the information above, select the best answer. Respond with ONLY the letter (A, B, C, or D):"""

# =============================================================================
# Baseline Prompts
# =============================================================================

BASELINE_MC_PROMPT = """Answer the following multiple-choice question about the image.

Question: {question}

Choices:
{choices}

Select the best answer and respond with ONLY the letter (A, B, C, or D):"""

BASELINE_DIRECT_PROMPT = """Answer the following question about the image in a few words.

Question: {question}

Answer:"""

# =============================================================================
# Helper Functions
# =============================================================================

def format_triage_prompt(question: str) -> str:
    """Format the triage prompt with the question"""
    return TRIAGE_PROMPT.format(question=question)


def format_visual_context_prompt(question: str) -> str:
    """Format the visual context extraction prompt"""
    return VISUAL_CONTEXT_PROMPT.format(question=question)


def format_query_generation_prompt(question: str, visual_context: str) -> str:
    """Format the search query generation prompt"""
    return QUERY_GENERATION_PROMPT.format(
        question=question,
        visual_context=visual_context
    )


def format_knowledge_agent_prompt(question: str, visual_context: str) -> str:
    """Format the knowledge agent prompt for LLM-based knowledge retrieval"""
    return KNOWLEDGE_AGENT_PROMPT.format(
        question=question,
        visual_context=visual_context
    )


def format_planner_prompt(question: str, context: str) -> str:
    """Format the query planner prompt"""
    return PLANNER_PROMPT.format(question=question, context=context)


def format_executor_prompt(question: str, context: str, choices: str = None) -> str:
    """Format the executor prompt"""
    if choices:
        return EXECUTOR_MC_PROMPT.format(
            question=question,
            context=context,
            choices=choices
        )
    else:
        return EXECUTOR_VISUAL_PROMPT.format(
            question=question,
            context=context
        )


def format_synthesizer_prompt(
    question: str,
    context: str,
    intermediate_answers: str,
    choices: str
) -> str:
    """Format the synthesizer prompt"""
    return SYNTHESIZER_PROMPT.format(
        question=question,
        context=context,
        intermediate_answers=intermediate_answers,
        choices=choices
    )


def format_baseline_prompt(question: str, choices: str = None) -> str:
    """Format the baseline prompt"""
    if choices:
        return BASELINE_MC_PROMPT.format(question=question, choices=choices)
    else:
        return BASELINE_DIRECT_PROMPT.format(question=question)
