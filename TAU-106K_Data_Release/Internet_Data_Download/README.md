## YouTube Videos Download

- Change the `paths` in the `.py` files to the specified paths (absolute paths are preferred).
- Download all YouTube videos.
    ```bash
    cd Internet_Data_Download 
    python YouTube/0_download_video.py
    ```
- Split the raw YouTube videos into video clips.
    ```bash
    python YouTube/1_extract_video.py
    ```
---

## BILIBILI Videos Download

- Change the `paths` in the `.py` files to the specified paths (absolute paths are preferred).
- Download all BILIBILI videos.
    ```bash
    cd Internet_Data_Download 
    python BILIBILI/0_download_video.py
    ```
- Split the raw BILIBILI videos into video clips.
    ```bash
    python BILIBILI/1_extract_video.py
    ```

---

## Notes
- To extract annotated images from YouTube and BILIBILI videos, please refer to `Data_Annotation/extract_images.py`.
- Due to the copyright issues, we cannot release videos from TikTok.