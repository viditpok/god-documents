/**
 * Node class used for implementing your linked data structures.
 *
 * DO NOT MODIFY THIS FILE!!
 *
 * @author CS 1332 TAs
 * @version 1.0
 */
public class LinkedNode<T> {

    private T data;
    private LinkedNode<T> next;

    /**
     * Constructs a new LinkedNode with the given data and next node reference.
     *
     * @param data the data stored in the new node
     * @param next the next node in the structure
     */
    LinkedNode(T data, LinkedNode<T> next) {
        this.data = data;
        this.next = next;
    }

    /**
     * Constructs a new LinkedNode with only the given data.
     *
     * @param data the data stored in the new node
     */
    LinkedNode(T data) {
        this(data, null);
    }

    /**
     * Gets the data.
     *
     * @return the data
     */
    T getData() {
        return data;
    }

    /**
     * Gets the next node.
     *
     * @return the next node
     */
    LinkedNode<T> getNext() {
        return next;
    }

    /**
     * Sets the next node.
     *
     * @param next the new next node
     */
    void setNext(LinkedNode<T> next) {
        this.next = next;
    }


    @Override
    public String toString() {
        return "Node containing: " + data;
    }
}