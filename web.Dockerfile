# The React client. Development image: it runs the Vite dev server, with the
# source bind-mounted, so an edit on the host is on screen without a rebuild.
FROM node:22-slim

WORKDIR /app

# Dependencies as their own layer, so editing a component does not reinstall
# node_modules.
COPY web/package.json web/package-lock.json* ./
RUN npm install

COPY web/ ./

EXPOSE 5173
CMD ["npm", "run", "dev"]
