//I worked on the homework assignment alone, using only course materials.

import java.util.ArrayList;
import java.util.NoSuchElementException;

/**
 * @author Vidit Pokharna
 * @version 1.0.0
 * @param <T> t
 **/
public class LinkedList<T> {
    /**
    *
    */
    private Node<T> head;
    /**
     *
     */
    private int size;

    /**
     *
     */
    public LinkedList() {
        head = null;
        size = 0;
    }

    /**
     * @param data t
     * @param index int
     */
    public void addAtIndex(T data, int index) {
        if (index >= 0 && index <= size) {
            Node<T> node = new Node<T>(data, null);
            if (index == 0) {
                node.setNext(head);
                head = node;
            } else {
                Node<T> now = head;
                int pos = 0;
                while (pos < index - 1) {
                    now = now.getNext();
                    pos++;
                }
                node.setNext(now.getNext());
                now.setNext(node);
            }
            size++;
        } else {
            throw new IllegalArgumentException("The index you have provided is invalid");
        }
    }

    /**
     * @param index int
     * @return t
     */
    public T removeFromIndex(int index) {
        if (index >= 0 && index < size) {
            if (index == 0) {
                Node<T> now = head;
                head = head.getNext();
                size -= 1;
                return now.getData();
            } else {
                Node<T> now = head;
                int position = 0;
                while (position < index - 1) {
                    now = now.getNext();
                    position += 1;
                }
                Node<T> temp = now.getNext();
                now.setNext(temp.getNext());
                size -= 1;
                return temp.getData();
            }
        } else {
            throw new IllegalArgumentException("The index you have provided is invalid");
        }
    }

    /**
     *
     */
    public void clear() {
        if (isEmpty()) {
            throw new NoSuchElementException("List is already empty");
        } else {
            head = null;
            size = 0;
        }
    }

    /**
     * @param index int
     * @return t
     */
    public T get(int index) {
        if (index >= 0 && index < size) {
            Node<T> now = head;
            int position = 0;
            while (position < index) {
                now = now.getNext();
                position++;
            }
            return now.getData();
        } else {
            throw new IllegalArgumentException("The index you have provided is invalid");
        }
    }

    /**
     * @return boolean
     */
    public boolean isEmpty() {
        return (head == null);
    }

    /**
     * @return arraylist
     */
    public ArrayList<T> toArrayList() {
        ArrayList<T> fin = new ArrayList<T>();
        Node<T> now = head;
        while (now != null) {
            fin.add(now.getData());
            now = now.getNext();
        }
        return fin;
    }

    /**
     * @return linkedlist
     */
    public LinkedList<String> fizzBuzzLinkedList() {
        LinkedList<String> fin = new LinkedList<>();
        Node<T> now = head;
        int position = 1;
        while (now != null) {
            if (position % 3 == 0 && position % 5 == 0) {
                fin.addAtIndex("FizzBuzz", position - 1);
            } else if (position % 3 == 0) {
                fin.addAtIndex("Fizz", position - 1);
            } else if (position % 5 == 0) {
                fin.addAtIndex("Buzz", position - 1);
            } else {
                fin.addAtIndex(position + ": " + now.getData().toString(), position - 1);
            }
            now = now.getNext();
            position += 1;
        }
        return fin;
    }
}
