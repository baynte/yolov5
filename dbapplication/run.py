from app import create_app
from flask_cors import CORS

flask_app = create_app()
CORS(flask_app, supports_credentials=True)

if __name__ == '__main__':
  flask_app.run('0.0.0.0', debug=True, port=5001)