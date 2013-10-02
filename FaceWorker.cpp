//g++ FaceWorker.cpp -o faceWorker

#include "getopt.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <istream>
#include <netdb.h>

int serverPort;
char *serverHost;
char *imageBasePath;

void error(char *msg) {
    perror(msg);
    exit(1);
}

void PrintUsage() {
    printf("\nworker client that runs the landmark detection code and the align-and-crop code\n");
    printf("Usage:  \n" \
            "   \t--basePath  [image file basepath... e.g. images/##.jpg]\n" \
            "   \t--clientMode [port]\n" \
            "   \t--remoteHost [hostname if not localhost]\n" \
            "\n");
}

void processOptions(int argc, char **argv){
    while (1){
        static struct option long_options[] = {
            {"help",        0, 0, 'h'}, 
            {"inputFace",   1, 0, 400},
            {"markedFace",  1, 0, 401},
            {"warpedFace",  1, 0, 402},
            {"croppedFace", 1, 0, 403},
            {"facePoints",  1, 0, 404},
            {"maskedFace",  1, 0, 405},
            {"maskedFace2", 1, 0, 406},
            
            {"clientMode",  1, 0, 500},
            {"remoteHost",  1, 0, 501},
            {"basePath",    1, 0, 502},

            {"faceList",    1, 0, 503},
            {"realignPoints",   1, 0, 504},
            {"idList",      1, 0, 505},
            
            {0,0,0,0} 
        };

        int option_index;;
        int c = getopt_long(argc, argv, "f:do:a:i:x",
                long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 'h':
                PrintUsage();
                exit(0);
                break;
                
            case 500:
                serverPort = atoi(optarg);
                break;
                
            case 501:
                serverHost = strdup(optarg);
                break;
                
            case 502:
                imageBasePath = strdup(optarg);
                break;

            default: 
                printf("Unrecognized option %d\n", c);
                break;
        }
    }
}

int detectFace(int faceId){
    printf("detectFace face #%d\n", faceId);

    char facePath[512];
    char pointPath[512];
    sprintf(facePath, "%s%d.jpg", imageBasePath, faceId);
    sprintf(pointPath, "%s%d_2.txt", imageBasePath, faceId); // TODO change to ../face_points/%d_2.txt"

    printf("\t%s \n\t%s \n", facePath, pointPath);
    
    char command[1024];
    sprintf(command, "/projects/grail/facegame/face-server/server/bin/detect-daniel.sh %s %s", facePath, pointPath);
    int exit_code = system(command);
    return exit_code;
}

int realignFace(int faceId){
    printf("Realigning face #%d\n", faceId);

    char facePath[512];
    char pointPath[512];
    char outPath[512];
    sprintf(facePath, "%s%d.jpg", imageBasePath, faceId);
    sprintf(pointPath, "%s%d_2.txt", imageBasePath, faceId); // TODO change to ../face_points/%d_2.txt"
    sprintf(outPath, "%s%d_", imageBasePath, faceId);
    printf("\t%s \n\t%s \n\t%s\n", facePath, pointPath, outPath);
    
    char command[1024];
    sprintf(command, "./alignFace %s %s %s", facePath, pointPath, outPath);
    int exit_code = system(command);
    return exit_code;
}

bool processReceivedMessage(char buffer [], int size, int sockfd) {
    printf("[ProcessReceivedMessage] Received message: %s\n", buffer);
    fflush(stdout);
    
    /* Parse the command */
    if (strncmp(buffer, "findFace: ", strlen("findFace: ")) == 0) {
        int faceId;
        sscanf(buffer + strlen("findFace: "), "%d", &faceId);
        printf("Processing face #%d\n", faceId);
        
        int success = 1;

        int detected_exit_code = detectFace(faceId);
        if (detected_exit_code == 0){
            success = realignFace(faceId);
        }

        char response[1024];
        
        if (success == 0){
            sprintf(response, "face %d: success\r\n", faceId);
        }
        else {
            sprintf(response, "face %d: failure\r\n", faceId);  
        }
        
        int n = write(sockfd, response, strlen(response));
        if (n < 0)
            error("[Worker] ERROR writing to socket");
        
        printf("[Worker] Finished processing face %d\n", faceId);

    }
    else if (strncmp(buffer, "realign: ", strlen("realign: ")) == 0) {
        int faceId;
        sscanf(buffer + strlen("realign: "), "%d", &faceId);

        int exit_code = realignFace(faceId);

        char response[1024];

        if (exit_code == 0){
            sprintf(response, "face %d realigned\r\n", faceId);
        }
        else{ 
            sprintf(response, "problem realigning face %d\r\n", faceId);
        }

        int n = write(sockfd, response, strlen(response));
        if (n < 0)
            error("[Worker] ERROR writing to socket");
        
        printf("[Worker] Finished processing face %d\n", faceId);

    }
    else if (strncmp(buffer, "shutdown ", strlen("shutdown")) == 0){
        printf("Received SHUTDOWN command.\n");
        return true;
    }   
    else {
        char *response = "unknown command :(\r\n";
        int n = write(sockfd, response, strlen(response));
        if (n < 0)
            error("[Worker] ERROR writing to socket");
    }
    
    fflush(stdout);
    return false;
    
}

void connectToServerAsWorker(){
    int sockfd, n;
    const int BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];
    struct sockaddr_in serv_addr;
    struct hostent *server;
    
    printf("[connectToServerAsWorker] Connecting to %s:%d\n", serverHost, serverPort);
    fflush(stdout);
    
    /* Set up socket stuff */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0){
        error("[connectToServerAsWorker] ERROR opening socket");
        printf("[connectToServerAsWorker] ERROR opening socket");
        fflush(stdout);
    }
    
    server = gethostbyname(serverHost);
    if (server == NULL) {
        fprintf(stderr,"[connectToServerAsWorker] ERROR, no such host\n");
        printf("[connectToServerAsWorker] ERROR, no such host\n");
        exit(0);
    }
    
    bzero((char *) &serv_addr, sizeof(serv_addr));
    
    serv_addr.sin_family = AF_INET;
    
    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    
    serv_addr.sin_port = htons(serverPort);
    
    /* Try to connect to server listening on host/port */
    if (connect(sockfd,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) < 0){
        error("[connectToServerAsWorker] ERROR connecting");
        printf("[connectToServerAsWorker] ERROR connecting\n");
        fflush(stdout);
    }
    
    bzero(buffer,BUFFER_SIZE);
    
    bool shutdown = false;
    
    printf("[connectToServerAsWorker] Connected!!\n");
    
    while(!shutdown && read(sockfd,buffer,BUFFER_SIZE-1)){
        shutdown = processReceivedMessage(buffer, BUFFER_SIZE-1, sockfd);
        bzero(buffer,BUFFER_SIZE);
    }
    
    printf("[connectToServerAsWorker] Shutting down...\n");
    
}



int main(int argc, char **argv){

    if (argc < 2 ){
        PrintUsage();
        exit(0);
    }

    serverPort = 0;
    serverHost = "localhost";
    processOptions(argc, argv);
    
    if (serverPort > 0){
        printf("\tserver port: %d, base path: [%s], host: [%s]\n", serverPort, imageBasePath, serverHost);
        connectToServerAsWorker();
    }

    return 0;
}