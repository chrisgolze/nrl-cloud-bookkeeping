import os
import openai
from google.cloud import storage, vision
from functions_framework import cloud_event

@cloud_event
def parse_file(cloud_event):
    try:
        bucket_name = cloud_event.data["bucket"]
        file_name = cloud_event.data["name"]

        if not file_name.endswith((".pdf", ".png", ".jpg", ".jpeg")):
            print("Skipped non-supported file.")
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

        print(f"✅ Processed and saved: {file_name}")

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        raise
