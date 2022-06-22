FROM golang:latest as build
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes build-essential git make gcc
WORKDIR /go/src/app
#add to copy local files into dockerfile and replace the demo app.go file with your own
COPY . .
RUN git config --global http.sslverify false
RUN git clone https://github.com/taufik-rama/fasttext-go-wrapper fastText
RUN git clone https://github.com/facebookresearch/fastText fb
RUN cd fb && make && cd .. && mkdir fastText/fastText/obj
RUN find fb -name '*.o' -exec mv {} fastText/fastText/obj \;
RUN mv fastText fastText2 && mv fastText2/* /go/src/app/
RUN cd fastText && make
#remove original app.go demo app to replace with your own file that has a main function
RUN rm app.go
RUN go mod download
RUN go build -o /go/bin/app
