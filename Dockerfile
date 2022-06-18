FROM debian:11-slim AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes build-essential git make gcc
RUN git config --global http.sslverify false
RUN git clone https://github.com/facebookresearch/fastText
RUN cd fastText && make && cd ..
RUN git clone https://github.com/DeanHnter/fasttext-go-wrapper
RUN cd fasttext-go-wrapper/fastText && make
RUN mkdir fastText/obj && \
    find . -name "*.o" -exec cp {} fasttext-go-wrapper/fastText/obj \;
RUN rm -r fastText && mv fasttext-go-wrapper/fastText fastText
