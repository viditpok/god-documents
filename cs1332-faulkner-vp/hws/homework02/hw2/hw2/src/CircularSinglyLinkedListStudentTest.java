import org.junit.Before;
import org.junit.Test;

import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class CircularSinglyLinkedListStudentTest {
    private static final int TIMEOUT = 200;
    private CircularSinglyLinkedList<Integer> list;

    @Before
    public void setUp() {
        list = new CircularSinglyLinkedList<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, list.size());
        assertNull(list.getHead());
    }

    @Test(timeout = TIMEOUT)
    public void testAddAtIndexException() {
        assertThrows(IllegalArgumentException.class, () -> {
            list.addAtIndex(0, null);
        });
        assertThrows(IndexOutOfBoundsException.class, () -> {
            list.addAtIndex(1, 7);
        });
        assertThrows(IndexOutOfBoundsException.class, () -> {
            list.addAtIndex(-1, 7);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testAddAtIndex() {
        list.addAtIndex(0,3 );
        list.addAtIndex(1, 4);
        list.addAtIndex(0, 1);
        list.addAtIndex(3, 5);

        assertEquals(4, list.size());
        Object[] arr = new Object[] {1, 3, 4, 5};

        assertArrayEquals(arr, list.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAddToFrontException() {
        assertThrows(IllegalArgumentException.class, () -> {
            list.addToFront(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testAddToFront() {
        list.addToFront(5);
        list.addToFront(4);
        list.addToFront(3);
        list.addToFront(1);

        assertEquals(4, list.size());
        Object[] arr = new Object[] {1, 3, 4, 5};

        assertArrayEquals(arr, list.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAddToBackException() {
        assertThrows(IllegalArgumentException.class, () -> {
            list.addToBack(null);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testAddToBack() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(4);
        list.addToBack(5);

        assertEquals(4, list.size());
        Object[] arr = new Object[] {1, 3, 4, 5};

        assertArrayEquals(arr, list.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFrontException() {
        assertThrows(NoSuchElementException.class, () -> {
            list.removeFromFront();
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveBackException() {
        assertThrows(NoSuchElementException.class, () -> {
            list.removeFromBack();
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveAtIndexException() {
        assertThrows(IndexOutOfBoundsException.class, () -> {
            list.removeAtIndex(-1);
        });
        assertThrows(IndexOutOfBoundsException.class, () -> {
            list.removeAtIndex(9);
        });
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveFront() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(4);
        list.addToBack(5);
        int i = list.size();
        while (!list.isEmpty()) {
            assertEquals(list.get(0), list.removeFromFront());
            assertEquals(list.size(), --i);
        }
        assertNull(list.getHead());
        assertEquals(list.size(), 0);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveBack() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(4);
        list.addToBack(5);
        int i = list.size();
        while (!list.isEmpty()) {
            assertEquals(list.get(i - 1), list.removeFromBack());
            assertEquals(list.size(), --i);
        }
        assertNull(list.getHead());
        assertEquals(list.size(), 0);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveIndex() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(4);
        list.addToBack(5);
        int i = list.size();
        while (!list.isEmpty()) {
            int n = (int) (Math.random() * list.size());
            assertEquals(list.get(n), list.removeAtIndex(n));
            assertEquals(list.size(), --i);
        }
        assertNull(list.getHead());
        assertEquals(list.size(), 0);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(1);
        list.addToBack(4);
        list.addToBack(5);
        list.addToBack(5);
        list.addToBack(7);
        Object[] arr = new Object[]{1,3,1,4,5,5,7};
        for (int i = 0; i < 100; ++i) {
            int n = (int) (Math.random() * list.size());
            assertEquals(arr[n], list.get(n));
        }
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveLastOccurrence() {
        list.addToBack(1);
        list.addToBack(3);
        list.addToBack(5);
        list.addToBack(4);
        list.addToBack(5);
        list.addToBack(5);
        list.addToBack(7);
        int n = list.size();
        Object[] arr = new Object[]{3,5,4,5,5,7};
        assertEquals((Object) 1, list.removeLastOccurrence(1));
        n--;
        assertEquals(n, list.size());
        assertArrayEquals(arr, list.toArray());

        arr = new Object[]{3,5,4,5,7};
        assertEquals((Object) 5, list.removeLastOccurrence(5));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        arr = new Object[]{3,5,4,5};
        assertEquals((Object) 7, list.removeLastOccurrence(7));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        arr = new Object[]{3,5,5};
        assertEquals((Object) 4, list.removeLastOccurrence(4));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        arr = new Object[]{3,5};
        assertEquals((Object) 5, list.removeLastOccurrence(5));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        assertThrows(NoSuchElementException.class, () -> {
            list.removeLastOccurrence(1);
        });

        arr = new Object[]{3};
        assertEquals((Object) 5, list.removeLastOccurrence(5));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        assertThrows(NoSuchElementException.class, () -> {
            list.removeLastOccurrence(1);
        });

        arr = new Object[]{};
        assertEquals((Object) 3, list.removeLastOccurrence(3));
        assertEquals(--n, list.size());
        assertArrayEquals(arr, list.toArray());

        list.clear();
        assertThrows(NoSuchElementException.class, () -> {
            list.removeLastOccurrence(1);
        });
    }
}