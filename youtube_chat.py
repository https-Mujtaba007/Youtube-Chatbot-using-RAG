
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from  langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough ,RunnableParallel
import streamlit as st
 
load_dotenv()
st.tile("Youtube RAG Chatbot")


video_id = st.text_input(
  "Enter Youtube Video ID"

)

quest= st.text_input("Ask a question")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
   task = "text-generation"
    
)
model= ChatHuggingFace(llm=llm)




if st.button("Ask Question"):
    try:
        # 1. Get transcript
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=['en','ur','hi'])

        transcript = " ".join([chunk.text for chunk in transcript_list])

        # 2. Split transcript into chunks (IMPORTANT PART)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([transcript])

        # 3. Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # 4. Vector DB
        vector_store = Chroma.from_documents(chunks, embeddings)

        # 5. Retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # 6. Format function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. Prompt
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer only from the provided transcript context.
If context is not enough, say you don't know.

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        # 8. LLM
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation"
        )

        model = ChatHuggingFace(llm=llm)

        # 9. Chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()

        final_chain = parallel_chain | prompt | model | parser

        # 10. Question
        answer = final_chain.invoke(quest)

        st.subheader("Answering")
        st.write(answer)

    except TranscriptsDisabled:
        st.error("No captions available for this video")
        st.stop()
