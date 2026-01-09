from openai import OpenAI


class LLMService:
    """Handles LLM API interactions for question answering.

    Attributes:
        client (OpenAI): The OpenAI client instance.
        model (str): The model identifier.
    """

    def __init__(self, api_key: str = "sk-f72baa81ab21443190dc16be46a2d8c7", base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        """Initializes the LLM service.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API endpoint.
            model (str): The model identifier. Defaults to "deepseek-chat".
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer based on the query and retrieved context.

        Args:
            query (str): The user's query.
            context (str): The retrieved context from the knowledge graph.

        Returns:
            str: The generated answer.
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge graph context.

Context:
{context}

User Query: {query}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
