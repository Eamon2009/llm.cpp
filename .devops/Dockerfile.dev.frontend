FROM node:20-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
