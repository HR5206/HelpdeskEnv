# run.py

from app.app_factory import create_app

# Create the app — used by both Flask dev server and gunicorn
app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )