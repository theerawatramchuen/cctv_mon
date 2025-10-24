import http.server
import socketserver
import os
import base64
import json
import urllib.parse
from http import cookies
import hashlib
import time
import mimetypes

# User credentials from the provided file
USER_CREDENTIALS = {
    "admin": "admin",
    "user": "password",
    "162395": "162395"
}

# Simple session management
sessions = {}

# Configuration
CONF_FOLDERS = ["conf_1", "conf_2", "conf_3", "conf_4", "conf_5"]
PORT = 8000

class ImageViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the main HTML page
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html_content().encode())
        elif self.path == '/check-auth':
            # Check if user is authenticated
            self.handle_check_auth()
        elif self.path.startswith('/list-images/'):
            # List images in a specific folder
            self.handle_list_images()
        elif self.path.startswith('/image/'):
            # Serve actual image files
            self.handle_serve_image()
        elif self.path == '/logout':
            self.handle_logout()
        else:
            # For any other path, show 404
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/login':
            self.handle_login()
        else:
            self.send_error(404)
    
    def handle_check_auth(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        # Check for session cookie
        cookie = cookies.SimpleCookie(self.headers.get('Cookie'))
        session_id = cookie.get('session_id')
        
        if session_id and session_id.value in sessions:
            response = {'loggedIn': True}
        else:
            response = {'loggedIn': False}
            
        self.wfile.write(json.dumps(response).encode())
    
    def handle_login(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        username = data.get('username', '')
        password = data.get('password', '')
        
        # Check credentials
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            # Create session
            session_id = hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()
            sessions[session_id] = {
                'username': username,
                'login_time': time.time()
            }
            
            # Set session cookie
            cookie = cookies.SimpleCookie()
            cookie['session_id'] = session_id
            cookie['session_id']['path'] = '/'
            cookie['session_id']['httponly'] = True
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Set-Cookie', cookie.output(header=''))
            self.end_headers()
            
            response = {'success': True}
            self.wfile.write(json.dumps(response).encode())
        else:
            # Invalid credentials
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {'success': False}
            self.wfile.write(json.dumps(response).encode())
    
    def handle_logout(self):
        cookie = cookies.SimpleCookie(self.headers.get('Cookie'))
        session_id = cookie.get('session_id')
        
        if session_id and session_id.value in sessions:
            del sessions[session_id.value]
        
        # Clear session cookie
        new_cookie = cookies.SimpleCookie()
        new_cookie['session_id'] = ''
        new_cookie['session_id']['path'] = '/'
        new_cookie['session_id']['expires'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
        
        self.send_response(200)
        self.send_header('Set-Cookie', new_cookie.output(header=''))
        self.end_headers()
        
        response = {'success': True}
        self.wfile.write(json.dumps(response).encode())
    
    def handle_list_images(self):
        # Check authentication first
        if not self.is_authenticated():
            self.send_error(401)
            return
        
        # Extract folder name from path
        folder_name = self.path.split('/')[-1]
        
        if folder_name not in CONF_FOLDERS:
            self.send_error(404, "Folder not found")
            return
        
        # Get list of image files in the folder
        image_files = []
        folder_path = os.path.join(os.getcwd(), folder_name)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and self.is_image_file(filename):
                    image_files.append({
                        'name': filename,
                        'path': f'/image/{folder_name}/{filename}',
                        'size': os.path.getsize(file_path)
                    })
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(image_files).encode())
    
    def handle_serve_image(self):
        # Check authentication first
        if not self.is_authenticated():
            self.send_error(401)
            return
        
        # Extract folder and filename from path
        path_parts = self.path.split('/')
        if len(path_parts) < 4:
            self.send_error(404)
            return
        
        folder_name = path_parts[2]
        filename = '/'.join(path_parts[3:])
        
        if folder_name not in CONF_FOLDERS:
            self.send_error(404)
            return
        
        # Serve the image file
        file_path = os.path.join(os.getcwd(), folder_name, filename)
        
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.send_error(404)
            return
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Length', str(len(file_data)))
            self.end_headers()
            self.wfile.write(file_data)
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")
    
    def is_authenticated(self):
        cookie = cookies.SimpleCookie(self.headers.get('Cookie'))
        session_id = cookie.get('session_id')
        return session_id and session_id.value in sessions
    
    def is_image_file(self, filename):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def get_html_content(self):
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Login Page Styles */
        #login-page {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            width: 100%;
        }
        
        .login-container {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        .login-title {
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: none;
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 16px;
        }
        
        .form-group input::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
        
        .login-btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(to right, #4776E6, #8E54E9);
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .error-message {
            color: #ff6b6b;
            text-align: center;
            margin-top: 15px;
            display: none;
        }
        
        /* Main App Styles */
        #app {
            display: none;
            width: 100%;
            height: 100vh;
            flex-direction: column;
            background: rgba(25, 25, 35, 0.95);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        .app-header {
            padding: 20px;
            background: rgba(40, 40, 60, 0.9);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .app-title {
            font-size: 24px;
            font-weight: 600;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .username {
            color: #8E54E9;
            font-weight: 500;
        }
        
        .logout-btn {
            padding: 8px 16px;
            background: linear-gradient(to right, #FF416C, #FF4B2B);
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .logout-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4);
        }
        
        .tabs-container {
            display: flex;
            background: rgba(30, 30, 45, 0.9);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            border-bottom: 3px solid transparent;
        }
        
        .tab:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tab.active {
            border-bottom: 3px solid #4776E6;
            background: rgba(255, 255, 255, 0.1);
        }
        
        .content-area {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            align-content: flex-start;
        }
        
        .thumbnail {
            width: 200px;
            height: 180px;
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .thumbnail:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        
        .thumbnail-img {
            width: 100%;
            height: 140px;
            object-fit: cover;
        }
        
        .thumbnail-title {
            padding: 8px;
            text-align: center;
            font-size: 14px;
            background: rgba(40, 40, 60, 0.9);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .no-images {
            text-align: center;
            padding: 40px;
            font-size: 18px;
            color: rgba(255, 255, 255, 0.7);
            width: 100%;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 18px;
            color: rgba(255, 255, 255, 0.7);
            width: 100%;
        }
        
        /* Modal for full-size image */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            max-width: 90%;
            max-height: 90%;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
        }
        
        .close-modal {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .close-modal:hover {
            color: #ff6b6b;
            transform: scale(1.1);
        }
        
        .image-info {
            position: absolute;
            bottom: 20px;
            left: 0;
            width: 100%;
            text-align: center;
            color: white;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
        }
    </style>
</head>
<body>
    <!-- Login Page -->
    <div id="login-page">
        <div class="login-container">
            <h2 class="login-title">Image Viewer Login</h2>
            <form id="login-form">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" placeholder="Enter your username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" placeholder="Enter your password" required>
                </div>
                <button type="submit" class="login-btn">Login</button>
                <div class="error-message" id="error-message">Invalid username or password</div>
            </form>
        </div>
    </div>

    <!-- Main App -->
    <div id="app">
        <div class="app-header">
            <div class="app-title">Image Viewer</div>
            <div class="user-info">
                <span class="username" id="username-display">User</span>
                <button class="logout-btn" id="logout-btn">Logout</button>
            </div>
        </div>
        <div class="tabs-container">
            <div class="tab" data-folder="conf_5">Clear Unzipping Act </div>
            <div class="tab" data-folder="conf_4">High Potential</div>
            <div class="tab" data-folder="conf_3">Moderate Potential</div>
            <div class="tab" data-folder="conf_2">Low Potential</div>
            <div class="tab active" data-folder="conf_1">No Potential</div>
        </div>
        <div class="content-area" id="content-area">
            <div class="loading">Loading images...</div>
        </div>
    </div>

    <!-- Modal for full-size image -->
    <div class="modal" id="image-modal">
        <span class="close-modal" id="close-modal">&times;</span>
        <img class="modal-content" id="modal-image">
        <div class="image-info" id="image-info"></div>
    </div>

    <script>
        // DOM Elements
        const loginPage = document.getElementById('login-page');
        const app = document.getElementById('app');
        const loginForm = document.getElementById('login-form');
        const errorMessage = document.getElementById('error-message');
        const logoutBtn = document.getElementById('logout-btn');
        const usernameDisplay = document.getElementById('username-display');
        const tabs = document.querySelectorAll('.tab');
        const contentArea = document.getElementById('content-area');
        const imageModal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const closeModal = document.getElementById('close-modal');
        const imageInfo = document.getElementById('image-info');
        
        // Current active tab
        let currentTab = 'conf_1';
        let currentUsername = '';
        
        // Login form submission
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            // Send login request to server
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Login successful
                    currentUsername = username;
                    usernameDisplay.textContent = username;
                    loginPage.style.display = 'none';
                    app.style.display = 'flex';
                    loadImages(currentTab);
                } else {
                    // Login failed
                    errorMessage.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                errorMessage.style.display = 'block';
            });
        });
        
        // Logout button
        logoutBtn.addEventListener('click', function() {
            // Send logout request to server
            fetch('/logout')
            .then(() => {
                app.style.display = 'none';
                loginPage.style.display = 'flex';
                loginForm.reset();
                errorMessage.style.display = 'none';
                currentUsername = '';
            });
        });
        
        // Tab switching
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Update active tab
                tabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Load images for the selected folder
                currentTab = this.getAttribute('data-folder');
                loadImages(currentTab);
            });
        });
        
        // Close modal
        closeModal.addEventListener('click', function() {
            imageModal.style.display = 'none';
        });
        
        // Close modal when clicking outside the image
        imageModal.addEventListener('click', function(e) {
            if (e.target === imageModal) {
                imageModal.style.display = 'none';
            }
        });
        
        // Function to load images for a folder
        function loadImages(folder) {
            contentArea.innerHTML = '<div class="loading">Loading images from ' + folder + '...</div>';
            
            fetch('/list-images/' + folder)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(images => {
                contentArea.innerHTML = '';
                
                if (images.length === 0) {
                    contentArea.innerHTML = '<div class="no-images">No images found in ' + folder + '</div>';
                    return;
                }
                
                images.forEach(image => {
                    const thumbnail = document.createElement('div');
                    thumbnail.className = 'thumbnail';
                    thumbnail.innerHTML = `
                        <img src="${image.path}" alt="${image.name}" class="thumbnail-img" loading="lazy">
                        <div class="thumbnail-title">${image.name}</div>
                    `;
                    
                    thumbnail.addEventListener('click', function() {
                        modalImage.src = image.path;
                        imageInfo.textContent = image.name + ' (' + formatFileSize(image.size) + ')';
                        imageModal.style.display = 'flex';
                    });
                    
                    contentArea.appendChild(thumbnail);
                });
            })
            .catch(error => {
                console.error('Error loading images:', error);
                contentArea.innerHTML = '<div class="no-images">Error loading images from ' + folder + '</div>';
            });
        }
        
        // Helper function to format file size
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        // Check if user is already logged in on page load
        fetch('/check-auth')
        .then(response => response.json())
        .then(data => {
            if (data.loggedIn) {
                // For simplicity, we'll just show the app but won't know the username
                // In a real app, you might want to store the username in the session
                loginPage.style.display = 'none';
                app.style.display = 'flex';
                loadImages(currentTab);
            }
        });
    </script>
</body>
</html>
"""

def run_server():
    with socketserver.TCPServer(("", PORT), ImageViewerHandler) as httpd:
        print(f"Image Viewer Server running at http://localhost:{PORT}")
        print("Available usernames: admin, user, 162395")
        print("Passwords are the same as usernames")
        print(f"Looking for folders: {CONF_FOLDERS}")
        
        # Check if conf folders exist
        for folder in CONF_FOLDERS:
            if os.path.exists(folder) and os.path.isdir(folder):
                print(f"✓ Found folder: {folder}")
            else:
                print(f"✗ Missing folder: {folder} - please create it")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_server()