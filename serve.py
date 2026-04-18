import http.server
import socketserver
import os
import re
import json
import ast

PORT = 8000

def get_config():
    """Extract configuration constants from testbench.py."""
    config = {
        "mgrs_tiles": ["31TDF"],
        "dates": ["2025/07/01"],
        "formats": ["webp"],
        "qualities": [75],
        "resampling": ["bilinear"],
        "exponents": [0.6],
        "scaling_ranges": [[0, 5000]]
    }
    try:
        if not os.path.exists("testbench.py"):
            return config

        with open("testbench.py", "r") as f:
            tree = ast.parse(f.read())
            
        target_keys = set(config.keys())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in target_keys:
                        try:
                            config[target.id.lower()] = ast.literal_eval(node.value)
                        except (ValueError, TypeError, SyntaxError):
                            pass
    except Exception as e:
        print(f"Error reading config: {e}")
    return config

class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def handle_range(self, is_head=False):
        """Shared logic for GET and HEAD Range requests."""
        if 'Range' not in self.headers:
            return super().do_GET() if not is_head else super().do_HEAD()

        range_header = self.headers.get('Range')
        match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if not match:
            return super().do_GET() if not is_head else super().do_HEAD()

        start_byte = int(match.group(1))
        end_byte_str = match.group(2)
        
        file_path = self.translate_path(self.path)
        if not os.path.isfile(file_path):
            return super().do_GET() if not is_head else super().do_HEAD()

        file_size = os.path.getsize(file_path)
        end_byte = int(end_byte_str) if end_byte_str else file_size - 1

        if start_byte >= file_size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return

        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(file_path))
        self.send_header('Content-Range', f'bytes {start_byte}-{end_byte}/{file_size}')
        self.send_header('Content-Length', str(end_byte - start_byte + 1))
        self.end_headers()

        if not is_head:
            with open(file_path, 'rb') as f:
                f.seek(start_byte)
                self.wfile.write(f.read(end_byte - start_byte + 1))

    def do_GET(self):
        if self.path == '/config':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(get_config()).encode())
            return
            
        self.handle_range(is_head=False)

    def do_HEAD(self):
        self.handle_range(is_head=True)

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), RangeRequestHandler) as httpd:
        print(f"Serving at http://127.0.0.1:{PORT} with Range support and CORS...")
        httpd.serve_forever()
