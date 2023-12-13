    import java.util.NoSuchElementException;

/**
 * Your implementation of a CircularSinglyLinkedList without a tail pointer.
 *
 * @author Vidit Pokharna
 * @version 1.0
 * @userid vpokharna3
 * @GTID 903772087
 *
 * Collaborators:
 *
 * Resources:
 */
public class CircularSinglyLinkedList<T> {

    /*
     * Do not add new instance variables or modify existing ones.
     */
    private CircularSinglyLinkedListNode<T> head;
    private int size;

    /*
     * Do not add a constructor.
     */

    /**
     * Adds the data to the specified index.
     *
     * Must be O(1) for indices 0 and size and O(n) for all other cases.
     *
     * @param index the index at which to add the new data
     * @param data  the data to add at the specified index
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index > size
     * @throws java.lang.IllegalArgumentException  if data is null
     */
    public void addAtIndex(int index, T data) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (index == 0) {
            addToFront(data);
        } else if (index == size) {
            addToBack(data);
        } else {
            CircularSinglyLinkedListNode<T> curr = head;
            int indice = 0;
            while (indice < index - 1) {
                curr = curr.getNext();
                indice++;
            }
            CircularSinglyLinkedListNode<T> newNode = new CircularSinglyLinkedListNode<T>(data);
            newNode.setNext(curr.getNext());
            curr.setNext(newNode);
            size++;
        }
    }

    /**
     * Adds the data to the front of the list.
     *
     * Must be O(1).
     *
     * @param data the data to add to the front of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToFront(T data) {
        if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (head == null) {
            CircularSinglyLinkedListNode<T> newNode = new CircularSinglyLinkedListNode<T>(data);
            head = newNode;
            head.setNext(head);
            size++;
        } else {
            CircularSinglyLinkedListNode<T> newNode = new CircularSinglyLinkedListNode<T>(head.getData());
            newNode.setNext(head.getNext());
            head.setNext(newNode);
            head.setData(data);
            size++;
        }
    }

    /**
     * Adds the data to the back of the list.
     *
     * Must be O(1).
     *
     * @param data the data to add to the back of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToBack(T data) {
        if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else {
            addToFront(data);
            head = head.getNext();
        }
    }

    /**
     * Removes and returns the data at the specified index.
     *
     * Must be O(1) for index 0 and O(n) for all other cases.
     *
     * @param index the index of the data to remove
     * @return the data formerly located at the specified index
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index >= size
     */
    public T removeAtIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else if (index == 0) {
            return removeFromFront();
        } else {
            CircularSinglyLinkedListNode<T> curr = head;
            CircularSinglyLinkedListNode<T> remove = null;
            int indice = 0;
            while (indice < index - 1) {
                curr = curr.getNext();
                indice++;
            }
            remove = curr.getNext();
            curr.setNext(curr.getNext().getNext());
            size--;
            return remove.getData();
        }
    }

    /**
     * Removes and returns the first data of the list.
     *
     * Must be O(1).
     *
     * @return the data formerly located at the front of the list
     * @throws java.util.NoSuchElementException if the list is empty
     */
    public T removeFromFront() {
        if (head == null) {
            throw new NoSuchElementException("The list is empty so no element can be removed from the linked list");
        } else if (size == 1) {
            T remove = head.getData();
            head = null;
            size--;
            return remove;
        } else {
            T remove = head.getData();
            head.setData(head.getNext().getData());
            head.setNext(head.getNext().getNext());
            size--;
            return remove;
        }
    }

    /**
     * Removes and returns the last data of the list.
     *
     * Must be O(n).
     *
     * @return the data formerly located at the back of the list
     * @throws java.util.NoSuchElementException if the list is empty
     */
    public T removeFromBack() {
        if (head == null) {
            throw new NoSuchElementException("The list is empty so no element can be removed from the linked list");
        } else if (size == 1) {
            T remove = head.getData();
            head = null;
            size--;
            return remove;
        } else {
            CircularSinglyLinkedListNode<T> curr = head;
            CircularSinglyLinkedListNode<T> remove = null;
            while (curr.getNext().getNext() != head) {
                curr = curr.getNext();
            }
            remove = curr.getNext();
            curr.setNext(head);
            size--;
            return remove.getData();
        }
    }

    /**
     * Returns the data at the specified index.
     *
     * Should be O(1) for index 0 and O(n) for all other cases.
     *
     * @param index the index of the data to get
     * @return the data stored at the index in the list
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index >= size
     */
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else {
            CircularSinglyLinkedListNode<T> curr = head;
            int indice = 0;
            while (indice < index) {
                curr = curr.getNext();
                indice++;
            }
            return curr.getData();
        }
    }

    /**
     * Returns whether or not the list is empty.
     *
     * Must be O(1).
     *
     * @return true if empty, false otherwise
     */
    public boolean isEmpty() {
        return (head == null);
    }

    /**
     * Clears the list.
     *
     * Clears all data and resets the size.
     *
     * Must be O(1).
     */
    public void clear() {
        head = null;
        size = 0;
    }

    /**
     * Removes and returns the last copy of the given data from the list.
     *
     * Do not return the same data that was passed in. Return the data that
     * was stored in the list.
     *
     * Must be O(n).
     *
     * @param data the data to be removed from the list
     * @return the data that was removed
     * @throws java.lang.IllegalArgumentException if data is null
     * @throws java.util.NoSuchElementException   if data is not found
     */
    public T removeLastOccurrence(T data) {
        if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (size == 0) {
            throw new NoSuchElementException("Through a traversal of the linked list, the data was not found");
        } else if (size == 1) {
            if (head.getData() == data) {
                T remove = head.getData();
                head = null;
                size--;
                return remove;
            } else {
                throw new NoSuchElementException("Through a traversal of the linked list, the data was not found");
            }
        } else {
            if (head.getData() == data) {
                T remove = head.getData();
                head.setData(head.getNext().getData());
                head.setNext(head.getNext().getNext());
                size--;
                return remove;
            }
            CircularSinglyLinkedListNode<T> curr = head.getNext();
            int index = 1;
            int index1 = -1;
            while (curr != head) {
                if (curr.getData() == data) {
                    index1 = index;
                }
                curr = curr.getNext();
                index++;
            }
            if (index1 == -1) {
                throw new NoSuchElementException("Through a traversal of the linked list, the data was not found");
            } else {
                return removeAtIndex(index1);
            }
        }
    }

    /**
     * Returns an array representation of the linked list.
     *
     * Must be O(n) for all cases.
     *
     * @return the array of length size holding all of the data (not the
     * nodes) in the list in the same order
     */
    public T[] toArray() {
        Object[] array1 = new Object[size];
        T[] array = (T[]) array1;
        if (head == null) {
            return array;
        } else if (head.getNext() == null) {
            array[0] = head.getData();
            return array;
        } else {
            array[0] = head.getData();
            CircularSinglyLinkedListNode<T> curr = head.getNext();
            int index = 1;
            while (curr != head) {
                array[index] = curr.getData();
                curr = curr.getNext();
                index++;
            }
            return array;
        }
    }

    /**
     * Returns the head node of the list.
     *
     * For grading purposes only. You shouldn't need to use this method since
     * you have direct access to the variable.
     *
     * @return the node at the head of the list
     */
    public CircularSinglyLinkedListNode<T> getHead() {
        // DO NOT MODIFY!
        return head;
    }

    /**
     * Returns the size of the list.
     *
     * For grading purposes only. You shouldn't need to use this method since
     * you have direct access to the variable.
     *
     * @return the size of the list
     */
    public int size() {
        // DO NOT MODIFY!
        return size;
    }
}
