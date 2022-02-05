mkdir -p <sub>/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > </sub>/.streamlit/config.toml