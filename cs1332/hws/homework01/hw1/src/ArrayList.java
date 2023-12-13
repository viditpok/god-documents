import java.util.NoSuchElementException;

/**
 * Your implementation of an ArrayList.
 *
 * @author Vidit Pokharna
 * @version 1.0
 * @userid vpokharna3
 * @GTID 903772087
 *
 * Collaborators: LIST ALL COLLABORATORS YOU WORKED WITH HERE
 *
 * Resources: LIST ALL NON-COURSE RESOURCES YOU CONSULTED HERE
 */
public class ArrayList<T> {

    /**
     * The initial capacity of the ArrayList.
     *
     * DO NOT MODIFY THIS VARIABLE!
     */
    public static final int INITIAL_CAPACITY = 9;

    // Do not add new instance variables or modify existing ones.
    private T[] backingArray;
    private int size;

    /**
     * Constructs a new ArrayList.
     *
     * Java does not allow for regular generic array creation, so you will have
     * to cast an Object[] to a T[] to get the generic typing.
     */
    public ArrayList() {
        backingArray = (T[]) (new Object[INITIAL_CAPACITY]);
    }

    /**
     * Adds the element to the specified index.
     *
     * Remember that this add may require elements to be shifted.
     *
     * Must be amortized O(1) for index size and O(n) for all other cases.
     *
     * @param index the index at which to add the new element
     * @param data  the data to add at the specified index
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index > size
     * @throws java.lang.IllegalArgumentException  if data is null
     */
    public void addAtIndex(int index, T data) {
        if (index > size || index < 0) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (size == backingArray.length) {
            Object[] tempArray = new Object[2 * backingArray.length];
            T[] newArray = (T[]) tempArray;
            for (int a = 0; a <= size; a++) {
                if (a < index) {
                    newArray[a] = backingArray[a];
                } else if (a == index) {
                    newArray[index] = data;
                } else {
                    newArray[a] = backingArray[a - 1];
                }
            }
            backingArray = newArray;
            size++;
        } else {
            for (int b = size - 1; b >= index; b--) {
                if (b < backingArray.length - 1) {
                    backingArray[b + 1] = backingArray[b];
                }
            }
            backingArray[index] = data;
            size++;
        }
    }

    /**
     * Adds the element to the front of the list.
     *
     * Remember that this add may require elements to be shifted.
     *
     * Must be O(n).
     *
     * @param data the data to add to the front of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToFront(T data) {
        if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (size == backingArray.length) {
            Object[] tempArray = new Object[2 * backingArray.length];
            T[] newArray = (T[]) tempArray;
            for (int a = 0; a < size; a++) {
                newArray[a + 1] = backingArray[a];
            }
            newArray[0] = data;
            backingArray = newArray;
            size++;
        } else if (size < backingArray.length) {
            for (int b = size - 1; b >= 0; b--) {
                backingArray[b + 1] = backingArray[b];
            }
            backingArray[0] = data;
            size++;
        }
    }

    /**
     * Adds the element to the back of the list.
     *
     * Must be amortized O(1).
     *
     * @param data the data to add to the back of the list
     * @throws java.lang.IllegalArgumentException if data is null
     */
    public void addToBack(T data) {
        if (data == null) {
            throw new IllegalArgumentException("The data provided does not have a value");
        } else if (size == backingArray.length) {
            Object[] tempArray = new Object[2 * backingArray.length];
            T[] newArray = (T[]) tempArray;
            for (int a = 0; a < size; a++) {
                newArray[a] = backingArray[a];
            }
            newArray[size] = data;
            backingArray = newArray;
            size++;
        } else if (size < backingArray.length) {
            backingArray[size] = data;
            size++;
        }
    }

    /**
     * Removes and returns the element at the specified index.
     *
     * Remember that this remove may require elements to be shifted.
     *
     * Must be O(1) for index size - 1 and O(n) for all other cases.
     *
     * @param index the index of the element to remove
     * @return the data formerly located at the specified index
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index >= size
     */
    public T removeAtIndex(int index) {
        if (index >= size || index < 0) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else {
            T extract = backingArray[index];
            for (int b = index; b < size - 1; b++) {
                backingArray[b] = backingArray[b + 1];
            }
            backingArray[size - 1] = null;
            size--;
            return extract;
        }
    }

    /**
     * Removes and returns the first element of the list.
     *
     * Remember that this remove may require elements to be shifted.
     *
     * Must be O(n).
     *
     * @return the data formerly located at the front of the list
     * @throws java.util.NoSuchElementException if the list is empty
     */
    public T removeFromFront() {
        if (size <= 0) {
            throw new NoSuchElementException("The list is empty and therefore no element can be removed");
        } else {
            T extract = backingArray[0];
            for (int b = 0; b < size - 1; b++) {
                backingArray[b] = backingArray[b + 1];
            }
            backingArray[size - 1] = null;
            size--;
            return extract;
        }
    }

    /**
     * Removes and returns the last element of the list.
     *
     * Must be O(1).
     *
     * @return the data formerly located at the back of the list
     * @throws java.util.NoSuchElementException if the list is empty
     */
    public T removeFromBack() {
        if (size <= 0) {
            throw new NoSuchElementException("The list is empty and therefore no element can be removed");
        } else {
            T extract = backingArray[size - 1];
            backingArray[size - 1] = null;
            size--;
            return extract;
        }
    }

    /**
     * Returns the element at the specified index.
     *
     * Must be O(1).
     *
     * @param index the index of the element to get
     * @return the data stored at the index in the list
     * @throws java.lang.IndexOutOfBoundsException if index < 0 or index >= size
     */
    public T get(int index) {
        if (index >= size || index < 0) {
            throw new IndexOutOfBoundsException("The index you have provided is outside the range of the array");
        } else {
            return backingArray[index];
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
        return size <= 0;
    }

    /**
     * Clears the list.
     *
     * Resets the backing array to a new array of the initial capacity and
     * resets the size.
     *
     * Must be O(1).
     */
    public void clear() {
        size = 0;
        Object[] array = new Object[INITIAL_CAPACITY];
        backingArray = (T[]) array;
    }

    /**
     * Returns the backing array of the list.
     *
     * For grading purposes only. You shouldn't need to use this method since
     * you have direct access to the variable.
     *
     * @return the backing array of the list
     */
    public T[] getBackingArray() {
        // DO NOT MODIFY THIS METHOD!
        return backingArray;
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
        // DO NOT MODIFY THIS METHOD!
        return size;
    }
}
