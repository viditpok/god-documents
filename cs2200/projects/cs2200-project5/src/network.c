#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>

#include "network.h"

/* Struct representing a UDP connection */
typedef struct _connection {
    int open;
    int socket_desc;
    struct addrinfo *host_info;
    struct _connection *next;
} connection_t;

/* List of connections */
static connection_t *c_list = NULL;

/* Create a connection and add it to the connections list */
static connection_t* make_connection(int socket_desc, struct addrinfo *host_info) {
    connection_t *conn = (connection_t*)malloc(sizeof(connection_t));
    conn->socket_desc = socket_desc;
    conn->host_info = host_info;
    conn->open = 1;
    
    conn->next = c_list;
    c_list = conn;
    
    return conn;
}

/* Get a connection from the connections list from its socket descriptor number */
static connection_t* get_connection(int socket_desc) {
    for (connection_t* conn = c_list; conn; conn = conn->next) {
        if (conn->socket_desc == socket_desc)
            return conn;
    }
    return NULL;
}

/*
 * Closes a connection in the connections list
 * Returns 0 on success, returns < 0 on failure
 */
static int close_connection(int socket_desc) {
    int r;

    for (connection_t* conn = c_list; conn; conn = conn->next) {
        if (conn->socket_desc == socket_desc) {
            r = close(socket_desc);
            if (r >= 0) conn->open = 0;
            return r;
        }
    }

    return -1;
}

/*
 * Remove a connection from the connections list and free it
 * Returns 0 on success, returns < 0 on failure
 */
static int remove_connection(int socket_desc) {
    connection_t* prev = NULL;
    for (connection_t* conn = c_list; conn; conn = conn->next) {
        if (conn->socket_desc == socket_desc) {
            if (conn == c_list)
                c_list = conn->next;
            else
                prev->next = conn->next;

            freeaddrinfo(conn->host_info);
            free(conn);
            return 0;
        }
        else
            prev = conn;
    }

    return -1;
}

/*
 * Opens a UDP connection to a given host on a given port
 * Returns:
 *  On success: an integer representing the connection handle
 *  Fails to convert host name to an address: -BAD_HOST
 *  Fails to create socket: -BAD_SOCKET
 */
int net_connect(char *host, int port) {
    // Set connection information
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;       // IPv4
    hints.ai_socktype = SOCK_DGRAM;  // UDP socket
    hints.ai_protocol = 0;           // IP protocol
    
    // Create a socket
    int socket_desc = socket(hints.ai_family, hints.ai_socktype, hints.ai_protocol);
    if (socket_desc < 0)
        return -BAD_SOCKET;

    // Set a timeout
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    if (setsockopt(socket_desc, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
        return -BAD_SOCKET;
    
    // Convert the host name to an address
    struct addrinfo *host_info;
    char port_str[10];
    sprintf(port_str, "%d", port);
    if (getaddrinfo(host, port_str, &hints, &host_info) < 0)
        return -BAD_HOST;
    
    // Create a connection
    make_connection(socket_desc, host_info);
    
    return socket_desc;
}

/*
 * Closes a UDP connection
 * Returns 0 on success, returns < 0 on failure
 */
int net_disconnect(int connection) {
    return close_connection(connection);
}

/*
 * Releases any memory associated with a connection.
 * Returns 0 on success, returns < 0 on failure
 */
int net_release(int connection) {
    return remove_connection(connection);
}

/*
 * Sends a packet across a UDP connection
 * Returns the size of the data sent
 */
int net_send_packet(int connection, packet_t *packet) {
    connection_t* conn = get_connection(connection);
    if (!conn->open)
        return -BAD_SOCKET;

    int r;
    while (1) {
        r = sendto(conn->socket_desc, packet, sizeof(packet_t), 0,
                   conn->host_info->ai_addr, conn->host_info->ai_addrlen);

        /* Return if bytes are successfully sent.
           If sendto is interrupted or times out, try again,
           else return the error. */
        if (r >= 0 || (r < 0 && ((errno != EAGAIN && errno != EINTR)
                                 || !conn->open))) return r;
    }
}

/*
 * Waits for a packet from a UDP connection,
 * and writes it to the provided packet buffer
 * Returns the size of the data received
 */
int net_recv_packet(int connection, packet_t *packet) {
    connection_t* conn = get_connection(connection);
    if (!conn->open)
        return -BAD_SOCKET;
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(struct sockaddr_in));
    socklen_t addrlen = sizeof(addr);

    int r;
    while (1) {
        r = recvfrom(conn->socket_desc, packet, sizeof(packet_t), 0,
                     (struct sockaddr*)(&addr), &addrlen);

        /* If recvfrom is interrupted or times out, try again,
           else return the error */
        if (r < 0 && ((errno != EAGAIN && errno != EINTR) || !conn->open)) return r;

        /* Validate the from address */
        if (addr.sin_addr.s_addr ==
            (*((struct sockaddr_in*)(conn->host_info->ai_addr))).sin_addr.s_addr) {
            return r;
        }
    }
}
