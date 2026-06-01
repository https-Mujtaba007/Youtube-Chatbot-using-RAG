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
st.title("Youtube RAG Chatbot")


video_id = st.text_input(
  "Enter Youtube Video ID"

)

quest= st.text_input("Ask a question")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
   task = "text-generation"
    
)
model= ChatHuggingFace(llm=llm)

# # video_id =  "Gfr50f6ZBvo"
# if st.button("Ask Question"):
#     try:
#         api = YouTubeTranscriptApi()
#         transcript_list = api.fetch(video_id,languages=['en'])

#         transcript =  " ".join([chunk.text for chunk in transcript_list]) 
# #        print(transcript)

#     except TranscriptsDisabled:
#         # print(" No captions availble for this video")
#         st.error("No Captions available")
#         st.stop()


#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         chunks = splitter.create_documents([transcript]) 
#         #  print(len(chunks))



# embeddings=  HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
# vector_store= Chroma.from_documents(chunks,embeddings)


# retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})
# # result =retriever.invoke('what is deepmind')
# # print(result)


# prompt= PromptTemplate(
#    template=  """
#      You are an helpful assistant.
#      Answer only from the provided transcript context.
#      If the context is insufficient ,just say you don't know.
#    {context}
#    Question : {question} """,
#    input_variables = {'context', 'question'}

# )

# question =  'Is the topic of aliens is discussed in this video? if yes hen what was discussed'
# # retrieved_doc = retriever.invoke(question)
# # print(retrieved_doc)

# # concatinated_text = "\n\n".join(doc.page_content for doc in retrieved_doc)

# # final_prompt= prompt.invoke({"context":concatinated_text, "question":question})

# # print(final_prompt)

# # result = model.invoke(final_prompt)
# # print(result.content)

# def format_docs(retrieved_doc):
#   concatinated_text = "\n\n".join(doc.page_content for doc in retrieved_doc)
#   return concatinated_text

# parallel_chain = RunnableParallel({ 
#   "context"  : retriever | RunnableLambda(format_docs),
#   "question" : RunnablePassthrough()
# })

# # chainer = parallel_chain.invoke('who is demis')
# # print(chainer)

# parser = StrOutputParser()

# finals = parallel_chain | prompt | model | parser
# # ending = finals.invoke('what is discussed about Ai in the video')
# answer = finals.invoke(question)
# # print(ending)
# st.subheader("Answer")
# st.write("answer")









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

        st.subheader("Answer")
        st.write(answer)

    except TranscriptsDisabled:
        st.error("No captions available for this video")
        st.stop()