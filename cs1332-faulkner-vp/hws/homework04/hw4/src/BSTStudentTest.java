import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

/**
 * This is a basic set of unit tests for BST.
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
public class BSTStudentTest {
    private static final int TIMEOUT = 200;
    private BST<Integer> tree;

    @Before
    public void setup() { tree = new BST<>(); };

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructor() {
        // empty list
        List<Integer> list = new ArrayList<>();
        tree = new BST<>(list);
        assertEquals(0, tree.size());

        // one item
        list.add(10);
        tree = new BST<>(list);
        assertEquals(1, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());

        // multiple items
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());

        // multiple items with duplicate
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        list.add(14);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testConstructorNull() {
        tree = new BST<>(null);
    }

    @Test(timeout = TIMEOUT)
    public void testAdd() {

        // add multiple items
        tree = new BST<>();
        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(17);
        tree.add(8);
        tree.add(12);
        tree.add(14);
        assertEquals(10, tree.size());
        assertEquals(10, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());

        // add multiple items with duplicate
        tree = new BST<>();
        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(17);
        tree.add(8);
        tree.add(12);
        tree.add(14);
        tree.add(14);
        tree.add(14);
        assertEquals(10, tree.size());
        assertEquals(10, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testAddNull() {
        tree = new BST<>();
        tree.add(null);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithNoChildren() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals((Integer) 2, tree.remove(2));
        assertEquals(9, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertNull(tree.getRoot().getLeft().getLeft());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithOneChild() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals((Integer) 7, tree.remove(7));
        assertEquals(9, tree.size());
        assertEquals((Integer) 10, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getData());
        assertNull(tree.getRoot().getLeft().getRight().getLeft());
        assertNull(tree.getRoot().getLeft().getRight().getRight());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 12, tree.getRoot().getRight().getLeft().getLeft().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithTwoChildren() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals((Integer) 10, tree.remove(10));
        assertEquals(9, tree.size());
        assertEquals((Integer) 12, tree.getRoot().getData());
        assertEquals((Integer) 5, tree.getRoot().getLeft().getData());
        assertEquals((Integer) 15, tree.getRoot().getRight().getData());
        assertEquals((Integer) 2, tree.getRoot().getLeft().getLeft().getData());
        assertEquals((Integer) 7, tree.getRoot().getLeft().getRight().getData());
        assertEquals((Integer) 13, tree.getRoot().getRight().getLeft().getData());
        assertEquals((Integer) 17, tree.getRoot().getRight().getRight().getData());
        assertEquals((Integer) 8, tree.getRoot().getLeft().getRight().getRight().getData());
        assertEquals((Integer) 14, tree.getRoot().getRight().getLeft().getRight().getData());
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemovEmptyTree() {
        tree = new BST<>();
        tree.remove(0);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveItemNotInTree() {
        tree = new BST<>();
        tree.add(10);
        tree.remove(5);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNull() {
        tree = new BST<>();
        tree.add(10);
        tree.remove(null);
    }

    @Test(timeout = TIMEOUT)
    public void testGet() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        assertSame(10, tree.get(10));
        assertSame(5, tree.get(5));
        assertSame(15, tree.get(15));
        assertSame(2, tree.get(2));
        assertSame(7, tree.get(7));
        assertSame(13, tree.get(13));
        assertSame(17, tree.get(17));
        assertSame(8, tree.get(8));
        assertSame(12, tree.get(12));
        assertSame(14, tree.get(14));
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetItemNotInTree() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        tree.get(11);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNull() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        tree = new BST<>(list);
        tree.get(null);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetEmptyTree() {
        tree = new BST<>();
        tree.get(10);
    }

    @Test(timeout = TIMEOUT)
    public void testContains() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        assertTrue(tree.contains(10));
        assertTrue(tree.contains(5));
        assertTrue(tree.contains(15));
        assertTrue(tree.contains(2));
        assertTrue(tree.contains(7));
        assertTrue(tree.contains(13));
        assertTrue(tree.contains(17));
        assertTrue(tree.contains(8));
        assertTrue(tree.contains(12));
        assertTrue(tree.contains(14));

        // data not in tree
        assertTrue(!tree.contains(0));
        assertTrue(!tree.contains(11));
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsNull() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);
        assertEquals(10, tree.size());
        tree.contains(null);
    }

    @Test(timeout = TIMEOUT)
    public void testPreorder() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);

        List<Integer> preorder = new ArrayList<>();
        preorder.add(10);
        preorder.add(5);
        preorder.add(2);
        preorder.add(7);
        preorder.add(8);
        preorder.add(15);
        preorder.add(13);
        preorder.add(12);
        preorder.add(14);
        preorder.add(17);

        assertEquals(preorder, tree.preorder());
    }

    @Test(timeout = TIMEOUT)
    public void testIndorder() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);

        List<Integer> inorder = new ArrayList<>();
        inorder.add(2);
        inorder.add(5);
        inorder.add(7);
        inorder.add(8);
        inorder.add(10);
        inorder.add(12);
        inorder.add(13);
        inorder.add(14);
        inorder.add(15);
        inorder.add(17);

        assertEquals(inorder, tree.inorder());
    }

    @Test(timeout = TIMEOUT)
    public void testPostorder() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);

        List<Integer> postorder = new ArrayList<>();
        postorder.add(2);
        postorder.add(8);
        postorder.add(7);
        postorder.add(5);
        postorder.add(12);
        postorder.add(14);
        postorder.add(13);
        postorder.add(17);
        postorder.add(15);
        postorder.add(10);

        assertEquals(postorder, tree.postorder());
    }

    @Test(timeout = TIMEOUT)
    public void testLevelorder() {
        List<Integer> list = new ArrayList<>();
        list.add(10);
        list.add(5);
        list.add(15);
        list.add(2);
        list.add(7);
        list.add(13);
        list.add(17);
        list.add(8);
        list.add(12);
        list.add(14);
        tree = new BST<>(list);

        List<Integer> levelorder = new ArrayList<>();
        levelorder.add(10);
        levelorder.add(5);
        levelorder.add(15);
        levelorder.add(2);
        levelorder.add(7);
        levelorder.add(13);
        levelorder.add(17);
        levelorder.add(8);
        levelorder.add(12);
        levelorder.add(14);

        assertEquals(levelorder, tree.levelorder());
    }

    @Test(timeout = TIMEOUT)
    public void testHeight() {
        // empty tree
        tree = new BST<>();
        assertEquals(-1, tree.height());

        // root
        tree.add(1);
        assertEquals(0, tree.height());

        // degenerate
        tree = new BST<>();
        tree.add(1);
        tree.add(2);
        tree.add(3);
        tree.add(4);
        assertEquals(3, tree.height());

        // larger tree
        tree = new BST<>();
        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(17);
        tree.add(8);
        tree.add(12);
        tree.add(14);
        assertEquals(3, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        tree = new BST<>();
        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(17);
        tree.add(8);
        tree.add(12);
        tree.add(14);
        assertEquals(10, tree.size());
        tree.clear();
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testKLargest() {
        tree = new BST<>();
        tree.add(10);
        tree.add(5);
        tree.add(15);
        tree.add(2);
        tree.add(7);
        tree.add(13);
        tree.add(17);
        tree.add(8);
        tree.add(12);
        tree.add(14);
        assertEquals(10, tree.size());

        List<Integer> largest = new ArrayList<>();
        largest.add(0, 17);
        assertEquals(largest, tree.kLargest(1)); // 17
        largest.add(0, 15);
        assertEquals(largest, tree.kLargest(2)); // 17, 15
        largest.add(0, 14);
        assertEquals(largest, tree.kLargest(3)); // 17, 15, 14
        largest.add(0, 13);
        assertEquals(largest, tree.kLargest(4)); // 17, 15, 14, 13
        largest.add(0, 12);
        assertEquals(largest, tree.kLargest(5)); // 17, 15, 14, 13, 12
        largest.add(0, 10);
        assertEquals(largest, tree.kLargest(6)); // 17, 15, 14, 13, 12, 10
        largest.add(0, 8);
        assertEquals(largest, tree.kLargest(7)); // 17, 15, 14, 13, 12, 10, 8
        largest.add(0, 7);
        assertEquals(largest, tree.kLargest(8)); // 17, 15, 14, 13, 12, 10, 8, 7
        largest.add(0, 5);
        assertEquals(largest, tree.kLargest(9)); // 17, 15, 14, 13, 12, 10, 8, 7, 5
        largest.add(0, 2);
        assertEquals(largest, tree.kLargest(10)); // 17, 15, 14, 13, 12, 10, 8, 7, 5, 2
    }


    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKLargestNegative() {
        tree = new BST<>();
        tree.add(10);
        tree.kLargest(-1);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKLargestOverSize() {
        tree = new BST<>();
        tree.add(10);
        tree.kLargest(2);
    }
}