FROM ollama/ollama

# Start the server in the background, wait until it accepts connections,
# then pull the model with the CLI.
# OLLAMA_HOST tells the CLI to connect to the local server.
# The RUN shell exiting kills the background server automatically;
# model weights in /root/.ollama are committed as a Docker layer.
RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 1; done && \
    ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
