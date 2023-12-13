#pragma once

#include <pthread.h>
#include "queue.h"

/*
 * A rtp connection.
 *
 * The rtp connection contains the threads for sending and receiving
 * packets as well as the different required locks. Additionally, it
 * holds the respective send and receive queues.
 */
typedef struct rtp_connection {
    int net_connection_handle;
    int alive;

    pthread_t recv_thread;
    pthread_t send_thread;

    /* FIX ME - insert any needed members */
    int ack_status;

    pthread_cond_t ack_cond;
    pthread_mutex_t ack_mutex;

    pthread_cond_t recv_cond;
    pthread_mutex_t recv_mutex;
    queue_t recv_queue;

    pthread_cond_t send_cond;
    pthread_mutex_t send_mutex;
    queue_t send_queue;

} rtp_connection_t;

/*  */
int checksum(char *buffer, int length);

/*  */
packet_t *packetize(char *buffer, int length, int *count);

/*  */
rtp_connection_t *rtp_connect(char *host, int port);

/*  */
int rtp_disconnect(rtp_connection_t *connection);

/*  */
int rtp_recv_message(rtp_connection_t *connection, char **buffer, int *length);

/*  */
int rtp_send_message(rtp_connection_t *connection, char *buffer, int length);
