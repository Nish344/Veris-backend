from flask import Blueprint, request, jsonify
import asyncio
import logging
import json
from graph import enhanced_run_investigation
import firebase_admin
from firebase_admin import credentials, firestore

investigate_bp = Blueprint('investigate', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase Admin SDK
# *** INSERT YOUR FIREBASE CONFIGURATION HERE ***
# Replace the placeholder with your Firebase service account credentials JSON file path
# For example, download the service account key JSON from Firebase Console > Project Settings > Service Accounts
cred = credentials.Certificate('cipher-quest-951dd-firebase-adminsdk-dvlsv-0bface6a6d.json')
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

def sanitize_for_firestore(data):
    """Recursively remove non-serializable types for Firestore."""
    try:
        if isinstance(data, dict):
            return {k: sanitize_for_firestore(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_for_firestore(v) for v in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)
    except Exception:
        pass

@investigate_bp.route('/investigate', methods=['POST'])
def start_investigation():
    """
    POST /investigate
    Starts a new investigation by calling enhanced_run_investigation and stores the result in Firestore.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'query' not in data or 'userid' not in data:
            return jsonify({'error': 'Missing query or userid in JSON data'}), 400
        
        query = data['query']
        userid = data['userid']
        
        # Run the investigation using enhanced_run_investigation from graph.py
        try:
            # Since Flask is synchronous, use asyncio.run to execute the async function
            investigation_data = asyncio.run(enhanced_run_investigation(query=query))
            investigation_id = investigation_data['investigation_id']
        except Exception as e:
            logging.error(f"Error running investigation: {str(e)}")
            return jsonify({
                'error': 'Investigation failed',
                'message': str(e)
            }), 500
        
        # Store investigation data in Firestore under users/<userid>/investigations/<investigation_id>
        try:
            doc_ref = db.collection('users').document(userid).collection('investigations').document(investigation_id)
            doc_ref.set(sanitize_for_firestore(investigation_data))
            logging.info(f"Stored investigation {investigation_id} in Firestore: users/{userid}/investigations/{investigation_id}")
        except Exception as e:
            logging.error(f"Error storing in Firestore: {str(e)}")
            return jsonify({
                'error': 'Failed to store in Firestore',
                'message': str(e)
            }), 500
        
        # Return response with investigation ID and data
        response = {
            'investigation_id': investigation_id,
            'investigation_data': investigation_data
        }
        logging.info(f"Completed investigation {investigation_id} for query: {query}, userid: {userid}")
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error in start_investigation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 200