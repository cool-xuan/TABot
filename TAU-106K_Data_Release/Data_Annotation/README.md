## Extract Annotated Images

- Change the `paths` in `extract_images.py` to the specified paths (absolute paths are preferred).
- Extract the images from the video clips (which are the output split videos of `1_extract_video.py`).
    ```bash
    python extract_images.py
    ```
---

## Generate Conversations

- Change the `input_dir`, `image_path`, `video_path` in `generate_image_conversation.py` and `generate_video_conversation.py`.
- Set the training epochs by setting `generation_freq`, and the number of epochs is set to `3` by default. 
- Generate all conversations.
    ```bash
    cd Data_Annotation
    python generate_video_conversation.py
    python generate_image_conversation.py
    ```

---

## Notes
- Please download and process the videos following `Internet_Data_Download` before generating the conversations.