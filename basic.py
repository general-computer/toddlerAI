import openai
import pinecone
import os

# OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Pinecone API Key
pinecone.init(api_key="YOUR_PINECONE_API_KEY")

# Create or connect to a Pinecone service
pinecone.deinit()
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
pinecone.create_index(index_name="embeddings", metric="cosine", shards=1)

pinecone.deinit()
pinecone.init(api_key="YOUR_PINECONE_API_KEY")

pinecone.create_index(index_name="embeddings", metric="cosine", shards=1)

# Function to generate response from OpenAI API
def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to store embeddings in Pinecone
def store_embedding(key, embedding):
    pinecone.upsert(index_name="embeddings", items={key: embedding})

# Function to get embeddings from Pinecone
def fetch_embedding(key):
    return pinecone.fetch(index_name="embeddings", ids=[key])[key]

# Main program
def main():
    try:
        while True:
            # Get input from the user
            user_input = input("Enter your query or reflection: ")

            # Generate a response using OpenAI's API
            response = generate_response(user_input)
            print("Response:", response)

            # Store the response's embedding in Pinecone
            store_embedding(user_input, response)

            # Reflection and criticism
            reflection = input("Enter your reflection on the response: ")
            criticism = input("Enter any criticism or feedback: ")

            # Next action steps
            next_action_prompt = f"Based on the reflection '{reflection}' and criticism '{criticism}', what should be the next action steps?"
            next_action = generate_response(next_action_prompt)
            print("Next Action Steps:", next_action)

    except KeyboardInterrupt:
        print("\nExiting program.")

    finally:
        # Clean up Pinecone resources
        pinecone.deinit()
        pinecone.init(api_key="YOUR_PINECONE_API_KEY")
        pinecone.delete_index(index_name="embeddings")

if __name__ == "__main__":
    main()

