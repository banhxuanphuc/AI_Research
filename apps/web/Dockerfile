FROM node:22-alpine AS deps
WORKDIR /app
# CA bổ sung (vd antivirus TLS scanning) - truyền qua additional_contexts "certs" trong compose
COPY --from=certs . /certs/
RUN cat /certs/*.crt > /certs/extra.pem 2>/dev/null || true
ENV NODE_EXTRA_CA_CERTS=/certs/extra.pem
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
