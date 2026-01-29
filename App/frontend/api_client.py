import requests
import config 

class BackendGateway:
    @staticmethod
    def generate_caption(prompt_str, json_data):
        """Sends user attributes to the backend to generate a text prompt."""
        payload = {"prompt_text": prompt_str, "json_data": json_data}
        try:
            resp = requests.post(f"{config.API_BASE_URL}/generate-caption", json=payload)
            if resp.status_code == 200:
                return resp.json().get("caption")
            else:
                return f"Error: {resp.text}"
        except Exception as e:
            return f"Connection Error: {e}"

    @staticmethod
    def generate_image(prompt_str):
        """Sends the text prompt to the backend to generate an image."""
        try:
            payload = {"sdxl_prompt": prompt_str}
            resp = requests.post(f"{config.API_BASE_URL}/generate-image", json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("image_base64")
            else:
                return None
        except Exception as e:
            print(f"API Error: {e}")
            return None