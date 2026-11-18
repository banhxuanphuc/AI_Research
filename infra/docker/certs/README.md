# CA bổ sung cho container (tuỳ máy, không commit)

Nếu mạng hoặc antivirus chặn và giải mã HTTPS (ví dụ Norton Web/Mail Shield), hãy đặt file root CA dạng PEM `*.crt` vào thư mục này. Image Python và Node sẽ nạp chúng vào trust store lúc build.

Thư mục trống (chỉ có README) thì build vẫn chạy bình thường. File `*.crt` đã được `.gitignore` loại trừ.
