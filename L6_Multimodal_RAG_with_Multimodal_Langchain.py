import lancedb

from langchain_core.runnables import (RunnableLambda, RunnableParallel, RunnablePassthrough)
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from PIL import Image
from utils import load_json_file


## ------------------------------------------------------ ##
LANCEDB_HOST_FILE = "./shared_data/.lancedb"

TBL_NAME = "test_tbl"
# TBL_NAME = "demo_tbl"

## ------------------------------------------------------ ##
embedder = BridgeTowerEmbeddings()

## ------------------------------------------------------ ##
vectorstore = MultimodalLanceDB(uri= LANCEDB_HOST_FILE, embedding= embedder, table_name= TBL_NAME)

retriever_module = vectorstore.as_retriever(search_type= 'similarity', search_kwargs= {"k" : 1})

## ------------------------------------------------------ ##
query = "What do the astronauts feel about their work?"
retrieved_video_segments = retriever_module.invoke(query)

retrieved_video_segment = retrieved_video_segments[0]

## ------------------------------------------------------ ##
retrieved_metadata = retrieved_video_segment.metadata['metadata']

frame_path = retrieved_metadata['extracted_frame_path']

transcript = retrieved_metadata['transcript']

video_path = retrieved_metadata['video_path']

timestamp = retrieved_metadata['mid_time_ms']

print(f"Transcript:\n{transcript}\n")
print(f"Path to extracted frame: {frame_path}")
print(f"Path to video: {video_path}")
print(f"Timestamp in ms when the frame was extracted: {timestamp}")
display(Image.open(frame_path))

## ------------------------------------------------------ ##
client = PredictionGuardClient()

lvlm_inference_module = LVLM(client= client)

## ------------------------------------------------------ ##
augmented_query_template = ("The transcript associated with the image is"
                            " '{transcript}'. {previous_query}")
augmented_query = augmented_query_template.format(transcript= transcript, previous_query= query, )
print(f"Augmented query is:\n{augmented_query}")

## ------------------------------------------------------ ##
input = {'prompt' : augmented_query, 'image' : frame_path}
response = lvlm_inference_module.invoke(input)

print('LVLM Response: ')
print(response)

## ------------------------------------------------------ ##
def prompt_processing(input):
    retrieved_results = input['retrieved_results']
    user_query = input['user_query']

    retrieved_result = retrieved_results[0]
    prompt_template = ("The transcript associated with the image is '{transcript}'. {user_query}")

    retrieved_metadata = retrieved_result.metadata['metadata']

    transcript = retrieved_metadata['transcript']

    frame_path = retrieved_metadata['extracted_frame_path']

    return {'prompt' : prompt_template.format(transcript= transcript, user_query= user_query),
            'image' : frame_path}


prompt_processing_module = RunnableLambda(prompt_processing)

## ------------------------------------------------------ ##
input_to_lvlm = prompt_processing_module.invoke({'retrieved_results' : retrieved_video_segments,
                                                'user_query' : query})

print(input_to_lvlm)

## ------------------------------------------------------ ##
mm_rag_chain = (RunnableParallel({"retrieved_results" : retriever_module,
                                "user_query" : RunnablePassthrough()})
                | prompt_processing_module | lvlm_inference_module)

## ------------------------------------------------------ ##
query1 = "What do the astronauts feel about their work?"
final_text_response1 = mm_rag_chain.invoke(query1)

print(f"USER Query: {query1}")
print(f"MM-RAG Response: {final_text_response1}")

## ------------------------------------------------------ ##
query2 = "What is the name of one of the astronauts?"
final_text_response2 = mm_rag_chain.invoke(query2)

print(f"USER Query: {query2}")
print(f"MM-RAG Response: {final_text_response2}")

## ------------------------------------------------------ ##
mm_rag_chain_with_retrieved_image = (RunnableParallel({"retrieved_results" : retriever_module,
                                                        "user_query" : RunnablePassthrough()})
                                    | prompt_processing_module
                                    | RunnableParallel({'final_text_output' : lvlm_inference_module,
                                                        'input_to_lvlm' : RunnablePassthrough()}) )

## ------------------------------------------------------ ##
response3 = mm_rag_chain_with_retrieved_image.invoke(query2)

print("Type of output of mm_rag_chain_with_retrieved_image is: ")
print(type(response3))
print(f"Keys of the dict are {response3.keys()}")

## ------------------------------------------------------ ##
final_text_response3 = response3['final_text_output']
path_to_extracted_frame = response3['input_to_lvlm']['image']

print(f"USER Query: {query2}")
print(f"MM-RAG Response: {final_text_response3}")
print("Retrieved frame: ")
display(Image.open(path_to_extracted_frame))

## ------------------------------------------------------ ##
query4 = "an astronaut's spacewalk"
response4 = mm_rag_chain_with_retrieved_image.invoke(query4)

final_text_response4 = response4['final_text_output']
path_to_extracted_frame4 = response4['input_to_lvlm']['image']

print(f"USER Query: {query4}")
print()
print(f"MM-RAG Response: {final_text_response4}")
print()
print("Retrieved frame: ")
display(Image.open(path_to_extracted_frame4))

## ------------------------------------------------------ ##
query5 = ("Describe the image of an astronaut's spacewalk with an amazing view of the earth "
        "from space behind")
response5 = mm_rag_chain_with_retrieved_image.invoke(query5)

final_text_response5 = response5['final_text_output']
path_to_extracted_frame5 = response5['input_to_lvlm']['image']

print(f"USER Query: {query5}")
print()
print(f"MM-RAG Response: {final_text_response5}")
print()
print("Retrieved Frame: ")
display(Image.open(path_to_extracted_frame5))

## ------------------------------------------------------ ##
query6 = ("An astronaut's spacewalk with an amazing view of the earth from space behind")
response6 = mm_rag_chain_with_retrieved_image.invoke(query6)

final_text_response6 = response6['final_text_output']
path_to_extracted_frame6 = response6['input_to_lvlm']['image']

print(f"USER Query: {query6}")
print()
print(f"MM-RAG Response: {final_text_response6}")
print()
print("Retrieved Frame: ")
display(Image.open(path_to_extracted_frame6))
