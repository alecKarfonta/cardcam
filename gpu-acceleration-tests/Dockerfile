FROM nginx:alpine

# Copy the test files
COPY . /usr/share/nginx/html/

# Copy nginx configuration and MIME types
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY mime.types /usr/share/nginx/html/mime.types

# Create log directories
RUN mkdir -p /var/log/nginx

# Expose port
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
