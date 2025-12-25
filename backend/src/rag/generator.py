#from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from backend.config.settings import Settings
import logging

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generate answers using FB15k-237 knowledge graph context"""
    
    def __init__(self, retriever, model_name: str = "meta-llama/llama-guard-4-12b", temperature: float = 0):
        self.retriever = retriever
        self.llm = ChatGroq(
                model_name=model_name,
                temperature=temperature,
                api_key=Settings().LLM_API_KEY
            )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions about a knowledge graph from Freebase.

                The knowledge graph contains entities and their relationships. Use the provided context to answer questions accurately.

                Context from Knowledge Graph:
                {context}

                Rules:
                - Only use information from the provided context
                - Be specific and mention entity names
                - If the context doesn't contain enough information, say so clearly
                - Format entity IDs like /m/xxx as readable names when possible"""),
            ("human", "{question}")
        ])
        
        self.output_parser = StrOutputParser()
    
    def generate(self, question: str, k: int = 10, depth: int = 2) -> Dict:
        """Generate answer for a question"""
        try:
            # Retrieve context
            context = self.retriever.retrieve_context(question, k, depth)
            
            # Generate answer
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )
            response = self.llm.invoke(messages)
            answer = self.output_parser.parse(response.content)
            
            return {
                'question': question,
                'answer': answer,
                'context': context,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'context': '',
                'status': 'error'
            }
    
    def generate_for_link_prediction(self, head: str, relation: str, 
                                     candidates: List[str]) -> str:
        """Generate tail entity prediction given head and relation"""
        context_parts = self.retriever.retrieve_for_link_prediction(head, relation)
        context = "\n".join(context_parts)
        
        prompt_text = f"""Given the head entity '{head}' and relation '{relation}',
                        predict the most likely tail entity from the following candidates:

                        Candidates:
                        {', '.join(candidates[:10])}

                        Knowledge graph context:
                        {context}

                        Answer with just the most likely entity ID."""
        
        messages = [("human", prompt_text)]
        response = self.llm.invoke(messages)
        
        return response.content.strip()