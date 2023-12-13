import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

import org.junit.Before;
import org.junit.Test;

/**
 * This is a basic set of unit tests for ArrayStack and LinkedStack.
 *
 * Passing these tests doesn't guarantee any grade on these assignments. These
 * student JUnits that we provide should be thought of as a sanity check to
 * help you get started on the homework and writing JUnits in general.
 *
 * We highly encourage you to write your own set of JUnits for each homework
 * to cover edge cases you can think of for each data structure. Your code must
 * work correctly and efficiently in all cases, which is why it's important
 * to write comprehensive tests to cover as many cases as possible.
 *
 * @author CS 1332 TAs
 * @version 1.0
 */
public class StackStudentTest {

    private static final int TIMEOUT = 200;
    private ArrayStack<String> array;
    private LinkedStack<String> linked;

    @Before
    public void setup() {
        array = new ArrayStack<>();
        linked = new LinkedStack<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, array.size());
        assertArrayEquals(new Object[ArrayStack.INITIAL_CAPACITY],
            array.getBackingArray());
        assertEquals(0, linked.size());
        assertNull(linked.getHead());
    }

    @Test(timeout = TIMEOUT)
    public void testArrayPush() {
        array.push("0a");   // 0a
        array.push("1a");   // 0a, 1a
        array.push("2a");   // 0a, 1a, 2a
        array.push("3a");   // 0a, 1a, 2a, 3a
        array.push("4a");   // 0a, 1a, 2a, 3a, 4a

        assertEquals(5, array.size());

        Object[] expected = new Object[ArrayStack.INITIAL_CAPACITY];
        expected[0] = "0a";
        expected[1] = "1a";
        expected[2] = "2a";
        expected[3] = "3a";
        expected[4] = "4a";
        assertArrayEquals(expected, array.getBackingArray());
    }

    @Test(timeout = TIMEOUT)
    public void testArrayPop() {
        String temp = "5a";

        array.push("0a");   // 0a
        array.push("1a");   // 0a, 1a
        array.push("2a");   // 0a, 1a, 2a
        array.push("3a");   // 0a, 1a, 2a, 3a
        array.push("4a");   // 0a, 1a, 2a, 3a, 4a
        array.push(temp);   // 0a, 1a, 2a, 3a, 4a, 5a
        assertEquals(6, array.size());

        assertSame(temp, array.pop());  // 0a, 1a, 2a, 3a, 4a

        assertEquals(5, array.size());

        Object[] expected = new Object[ArrayStack.INITIAL_CAPACITY];
        expected[0] = "0a";
        expected[1] = "1a";
        expected[2] = "2a";
        expected[3] = "3a";
        expected[4] = "4a";
        assertArrayEquals(expected, array.getBackingArray());
    }

    @Test(timeout = TIMEOUT)
    public void testArrayPeek() {
        String temp = "4a";

        array.push("0a");   // 0a
        array.push("1a");   // 0a, 1a
        array.push("2a");   // 0a, 1a, 2a
        array.push("3a");   // 0a, 1a, 2a, 3a
        array.push(temp);   // 0a, 1a, 2a, 3a, 4a
        assertEquals(5, array.size());

        assertSame(temp, array.peek());
    }

    @Test(timeout = TIMEOUT)
    public void testLinkedPush() {
        linked.push("0a");  // 0a
        linked.push("1a");  // 1a, 0a
        linked.push("2a");  // 2a, 1a 0a
        linked.push("3a");  // 3a, 2a, 1a 0a
        linked.push("4a");  // 4a, 3a, 2a, 1a 0a

        assertEquals(5, linked.size());

        LinkedNode<String> cur = linked.getHead();
        assertNotNull(cur);
        assertEquals("4a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("3a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("2a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("1a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("0a", cur.getData());

        cur = cur.getNext();
        assertNull(cur);
    }

    @Test(timeout = TIMEOUT)
    public void testLinkedPop() {
        String temp = "5a";

        linked.push("0a");  // 0a
        linked.push("1a");  // 1a, 0a
        linked.push("2a");  // 2a, 1a, 0a
        linked.push("3a");  // 3a, 2a, 1a, 0a
        linked.push("4a");  // 4a, 3a, 2a, 1a, 0a
        linked.push(temp);  // 5a, 4a, 3a, 2a, 1a, 0a
        assertEquals(6, linked.size());

        assertSame(temp, linked.pop()); // 4a, 3a, 2a, 1a, 0a

        assertEquals(5, linked.size());

        LinkedNode<String> cur = linked.getHead();
        assertNotNull(cur);
        assertEquals("4a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("3a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("2a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("1a", cur.getData());

        cur = cur.getNext();
        assertNotNull(cur);
        assertEquals("0a", cur.getData());

        cur = cur.getNext();
        assertNull(cur);
    }

    @Test(timeout = TIMEOUT)
    public void testLinkedPeek() {
        String temp = "4a";

        linked.push("0a");  // 0a
        linked.push("1a");  // 1a, 0a
        linked.push("2a");  // 2a, 1a, 0a
        linked.push("3a");  // 3a, 2a, 1a, 0a
        linked.push(temp);  // 4a, 3a, 2a, 1a, 0a
        assertEquals(5, linked.size());

        assertSame(temp, linked.peek());
    }
}
