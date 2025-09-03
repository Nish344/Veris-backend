from flask import Flask
from flask_cors import CORS
from routes.investigate import investigate_bp
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)
    
    # # Configuration
    # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    # app.config['MEDIA_DIRECTORIES'] = {
    #     'screenshots': 'media/screenshots',
    #     'pictures': 'media/pictures'
    # }
    # 
    # # Ensure media directories exist
    # for media_type, path in app.config['MEDIA_DIRECTORIES'].items():
    #     os.makedirs(path, exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(investigate_bp, url_prefix='/api')
    # app.register_blueprint(status_bp, url_prefix='/api')
    # app.register_blueprint(media_bp, url_prefix='/api')
    
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

