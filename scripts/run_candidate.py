"""Candidate-only secret mapping; the serving/model implementation is unchanged."""
import os

os.environ['OPEN_JEV_API_KEY'] = os.environ.pop('OPEN_JEV_CANDIDATE_API_KEY')
os.execvp('uvicorn', ['uvicorn', 'server:create_app', '--factory', '--host',
    '0.0.0.0', '--port', '8080', '--workers', '1', '--no-access-log',
    '--limit-concurrency', '16', '--timeout-keep-alive', '5'])
