import os
import functions_framework
from google.cloud import storage, vision
import openai

@functions_framework.cloud_event
def parse_file(cloud_event):
    bucket_name = cloud_event.data["bucket"]
    file_name = cloud_event.data["name"]

    if not file_name.endswith((".pdf", ".png", ".jpg", ".jpeg")):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    contents = blob.download_as_bytes()

    vision_client = vision.ImageAnnotatorClient()
    image = vision.Image(content=contents)
    response = vision_client.document_text_detection(image=image)
    raw_text = response.full_text_annotation.text

    openai.api_key = os.environ["OPENAI_API_KEY"]
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": raw_text}],
        temperature=0.2
    )

    result = gpt_response.choices[0].message.content
    output_blob = client.bucket("nrl-parsed").blob(f"cleaned/{file_name}.txt")
    output_blob.upload_from_string(result)

    print(f"Processed and saved: {file_name}")
