import os
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000

def start_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Server started at http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()

if __name__ == '__main__':
    # Start the server in a background thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Automatically open the default web browser to the dashboard
    url = f"http://localhost:{PORT}"
    print(f"Opening dashboard in your browser: {url}")
    webbrowser.open(url)
    
    # Keep the script running until the user stops it
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down the dashboard server...")
