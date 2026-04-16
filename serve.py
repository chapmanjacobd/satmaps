import http.server
import socketserver
import os
import re

PORT = 8000

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
        self.handle_range(is_head=False)

    def do_HEAD(self):
        self.handle_range(is_head=True)

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), RangeRequestHandler) as httpd:
        print(f"Serving at http://127.0.0.1:{PORT} with Range support and CORS...")
        httpd.serve_forever()
