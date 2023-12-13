#pragma once

/* Maximum size of the payload */
#define MAX_PAYLOAD_LENGTH 10

/* List of possible packet types */
enum {
    DATA = 0,
    LAST_DATA,
    ACK,
    NACK,
    TERM
};

/* Error codes */
enum {
    BAD_HOST = 1,
    BAD_SOCKET
};

/*
 * A packet.
 * 
 * This is the necessary form to send data through the network.
 * 
 * The packet struct holds the type of the packet, its calculated checksum,
 * the size of the data being sent, and the data itself. 
 */
typedef struct packet {
    int type;
    int checksum;
    int payload_length;
    char payload[MAX_PAYLOAD_LENGTH];
} packet_t;

/*
 * Opens a UDP connection to a given host on a given port
 * Returns:
 *  On success: an integer representing the connection handle
 *  Fails to convert host name to an address: -BAD_HOST
 *  Fails to create socket: -BAD_SOCKET
 */
int net_connect(char *host, int port);

/*
 * Closes a UDP connection
 * Returns 0 on success, returns < 0 on failure
 */
int net_disconnect(int connection);

/*
 * Releases any memory and resources associated with a connection.
 * Returns 0 on success, returns < 0 on failure
 */
int net_release(int connection);

/*
 * Sends a packet across a UDP connection
 * Returns the size of the data sent
 */
int net_send_packet(int connection, packet_t *packet);

/*
 * Waits for a packet from a UDP connection,
 * and writes it to the provided packet buffer
 * Returns the size of the data received
 */
int net_recv_packet(int connection, packet_t *packet);
