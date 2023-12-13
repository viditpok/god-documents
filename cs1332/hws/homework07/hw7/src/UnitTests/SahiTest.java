import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import org.junit.Before;
import org.junit.Test;

/**
 * @author Sahithya Pasagada
 * @version 1.0
 */

//Note that get(), contains(), and height() test cases are throughout the add and remove cases.

public class SahiTest {
    private static final int TIMEOUT = 200;
    private AVL<Integer> tree;

    @Before
    public void setup() {
        tree = new AVL<>();
    }

    @Test(timeout = TIMEOUT)
    public void testInitialization() {
        assertEquals(0, tree.size());
        assertNull(tree.getRoot());
    }

    @Test(timeout = TIMEOUT)
    public void testConstructor() {
        List<Integer> elements = new ArrayList<>();
        elements.add(23);
        elements.add(43);
        elements.add(13);
        elements.add(2);
        elements.add(100);
        elements.add(43);
        elements.add(43);
        elements.add(12);

        tree = new AVL<>(elements);

        assertTrue(tree.contains(23));
        assertTrue(tree.contains(2));
        assertTrue(tree.contains(100));
        assertTrue(tree.contains(12));
        assertFalse(tree.contains(1000));

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {23, 12, 2, 13, 43, 100};
        Integer[] expectedBFS = {0, 0, 0, 0, -1, 0};
        Integer[] expectedHeights = {2, 1, 0, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

        assertEquals(6, tree.size());

        List<Integer> deepest = new ArrayList<>();
        deepest.add(23);
        deepest.add(12);
        deepest.add(2);
        deepest.add(13);
        deepest.add(43);
        deepest.add(100);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();
        sorted.add(12);
        sorted.add(13);
        sorted.add(23);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(2, 43).toArray());

        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(2, 12).toArray());
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(43, 100).toArray());

        sorted.add(43);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(23, 100).toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testAddRoot() {
        tree.add(8);
        assertEquals(1, tree.size());
        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 8, root.getData());
        assertEquals(0, root.getHeight());
        assertEquals(0, root.getBalanceFactor());

        // checks one node deepest and sorted
        List<Integer> deepest = new ArrayList<>();
        deepest.add(8);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();
        sorted.add(8);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(7, 9).toArray());

        sorted.remove(0);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 9).toArray());
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 8).toArray());
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(6, 7).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAddNoRotations() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertSame((Integer) 80, tree.get(80));
        assertSame((Integer) 30, tree.get(30));
        assertSame((Integer) 50, tree.get(50));

        assertTrue(tree.contains(60));
        assertTrue(tree.contains(20));
        assertTrue(tree.contains(90));
        assertTrue(tree.contains(40));
        assertFalse(tree.contains(21));

        assertEquals(7, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 30, 20, 40, 80, 60, 90};
        Integer[] expectedBFS = {0, 0, 0, 0, 0, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testAddRightLefttRotation() {
        tree.add(14);
        tree.add(12);
        tree.add(34);
        tree.add(24);
        tree.add(32);

        assertSame((Integer) 14, tree.get(14));
        assertSame((Integer) 12, tree.get(12));
        assertSame((Integer) 32, tree.get(32));

        assertEquals(5, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 14, root.getData());
        assertEquals(2, root.getHeight());
        assertEquals(-1, root.getBalanceFactor());

        AVLNode<Integer> left = root.getLeft();
        assertEquals((Integer) 12, left.getData());
        assertEquals(0, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> right = root.getRight();
        assertEquals((Integer) 32, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        AVLNode<Integer> rightLeft = right.getLeft();
        assertEquals((Integer) 24, rightLeft.getData());
        assertEquals(0, rightLeft.getHeight());
        assertEquals(0, rightLeft.getBalanceFactor());

        AVLNode<Integer> rightRight = right.getRight();
        assertEquals((Integer) 34, rightRight.getData());
        assertEquals(0, rightRight.getHeight());
        assertEquals(0, rightRight.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddRightRotation() {
        tree.add(20);
        tree.add(3);
        tree.add(27);
        tree.add(2);
        tree.add(9);

        assertEquals(5, tree.size());

        tree.add(16);
        assertEquals(6, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 9, root.getData());
        assertEquals(2, root.getHeight());
        assertEquals(0, root.getBalanceFactor());

        AVLNode<Integer> left = root.getLeft();
        assertEquals((Integer) 3, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(1, left.getBalanceFactor());

        AVLNode<Integer> leftOne = left.getLeft();
        assertEquals((Integer) 2, leftOne.getData());
        assertEquals(0, leftOne.getHeight());
        assertEquals(0, leftOne.getBalanceFactor());

        AVLNode<Integer> right = root.getRight();
        assertEquals((Integer) 20, right.getData());
        assertEquals(1, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        AVLNode<Integer> rightLeft = right.getLeft();
        assertEquals((Integer) 16, rightLeft.getData());
        assertEquals(0, rightLeft.getHeight());
        assertEquals(0, rightLeft.getBalanceFactor());

        AVLNode<Integer> rightRight = right.getRight();
        assertEquals((Integer) 27, rightRight.getData());
        assertEquals(0, rightRight.getHeight());
        assertEquals(0, rightRight.getBalanceFactor());

    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRightRotation() {
        tree.add(20);
        tree.add(3);
        tree.add(27);
        tree.add(2);
        tree.add(9);
        tree.add(22);
        tree.add(32);
        tree.add(1);
        tree.add(6);
        tree.add(13);

        assertEquals(10, tree.size());

        assertTrue(tree.contains(20));
        assertTrue(tree.contains(32));
        assertTrue(tree.contains(1));
        assertTrue(tree.contains(27));
        assertFalse(tree.contains(21));

        tree.add(16);
        assertEquals(11, tree.size());

        assertSame((Integer) 16, tree.get(16));
        assertSame((Integer) 27, tree.get(27));
        assertSame((Integer) 1, tree.get(1));

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 9, root.getData());
        assertEquals(3, root.getHeight());
        assertEquals(0, root.getBalanceFactor());

        AVLNode<Integer> left = root.getLeft();
        assertEquals((Integer) 3, left.getData());
        assertEquals(2, left.getHeight());
        assertEquals(1, left.getBalanceFactor());

        AVLNode<Integer> leftLeft = left.getLeft();
        assertEquals((Integer) 2, leftLeft.getData());
        assertEquals(1, leftLeft.getHeight());
        assertEquals(1, leftLeft.getBalanceFactor());

        AVLNode<Integer> leftLeftLeft = leftLeft.getLeft();
        assertEquals((Integer) 1, leftLeftLeft.getData());
        assertEquals(0, leftLeftLeft.getHeight());
        assertEquals(0, leftLeftLeft.getBalanceFactor());

        AVLNode<Integer> leftRight = left.getRight();
        assertEquals((Integer) 6, leftRight.getData());
        assertEquals(0, leftRight.getHeight());
        assertEquals(0, leftRight.getBalanceFactor());

        AVLNode<Integer> right = root.getRight();
        assertEquals((Integer) 20, right.getData());
        assertEquals(2, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        AVLNode<Integer> rightLeft = right.getLeft();
        assertEquals((Integer) 13, rightLeft.getData());
        assertEquals(1, rightLeft.getHeight());
        assertEquals(-1, rightLeft.getBalanceFactor());

        AVLNode<Integer> rightLeftRight = rightLeft.getRight();
        assertEquals((Integer) 16, rightLeftRight.getData());
        assertEquals(0, rightLeftRight.getHeight());
        assertEquals(0, rightLeftRight.getBalanceFactor());

        AVLNode<Integer> rightRight = right.getRight();
        assertEquals((Integer) 27, rightRight.getData());
        assertEquals(1, rightRight.getHeight());
        assertEquals(0, rightRight.getBalanceFactor());

        AVLNode<Integer> rightRightLeft = rightRight.getLeft();
        assertEquals((Integer) 22, rightRightLeft.getData());
        assertEquals(0, rightRightLeft.getHeight());
        assertEquals(0, rightRightLeft.getBalanceFactor());

        AVLNode<Integer> rightRightRight = rightRight.getRight();
        assertEquals((Integer) 32, rightRightRight.getData());
        assertEquals(0, rightRightRight.getHeight());
        assertEquals(0, rightRightRight.getBalanceFactor());

    }

    @Test(timeout = TIMEOUT)
    public void testAddLeftRotation() {
        tree.add(23);
        tree.add(12);
        tree.add(45);
        tree.add(13);
        tree.add(15);

        assertEquals(5, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 23, root.getData());
        assertEquals(2, root.getHeight());
        assertEquals(1, root.getBalanceFactor());

        AVLNode<Integer> left = root.getLeft();
        assertEquals((Integer) 13, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> leftLeft = left.getLeft();
        assertEquals((Integer) 12, leftLeft.getData());
        assertEquals(0, leftLeft.getHeight());
        assertEquals(0, leftLeft.getBalanceFactor());

        AVLNode<Integer> leftRight = left.getRight();
        assertEquals((Integer) 15, leftRight.getData());
        assertEquals(0, leftRight.getHeight());
        assertEquals(0, leftRight.getBalanceFactor());

        AVLNode<Integer> right = root.getRight();
        assertEquals((Integer) 45, right.getData());
        assertEquals(0, right.getHeight());
        assertEquals(0, right.getBalanceFactor());
    }

    @Test(timeout = TIMEOUT)
    public void testAddMultipleRotations1() {
        tree.add(50);
        tree.add(40);
        tree.add(55);
        tree.add(35);
        tree.add(45);
        tree.add(34);
        tree.add(36);
        tree.add(44);
        tree.add(46);
        tree.add(48);

        assertSame((Integer) 40, tree.get(40));
        assertSame((Integer) 44, tree.get(44));
        assertSame((Integer) 48, tree.get(48));

        assertEquals(10, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 40, root.getData());
        assertEquals(3, root.getHeight());
        assertEquals(-1, root.getBalanceFactor());

        AVLNode<Integer> left = root.getLeft();
        assertEquals((Integer) 35, left.getData());
        assertEquals(1, left.getHeight());
        assertEquals(0, left.getBalanceFactor());

        AVLNode<Integer> leftLeft = left.getLeft();
        assertEquals((Integer) 34, leftLeft.getData());
        assertEquals(0, leftLeft.getHeight());
        assertEquals(0, leftLeft.getBalanceFactor());

        AVLNode<Integer> leftRight = left.getRight();
        assertEquals((Integer) 36, leftRight.getData());
        assertEquals(0, leftRight.getHeight());
        assertEquals(0, leftRight.getBalanceFactor());

        AVLNode<Integer> right = root.getRight();
        assertEquals((Integer) 46, right.getData());
        assertEquals(2, right.getHeight());
        assertEquals(0, right.getBalanceFactor());

        AVLNode<Integer> rightLeft = right.getLeft();
        assertEquals((Integer) 45, rightLeft.getData());
        assertEquals(1, rightLeft.getHeight());
        assertEquals(1, rightLeft.getBalanceFactor());

        AVLNode<Integer> rightLeftLeft = rightLeft.getLeft();
        assertEquals((Integer) 44, rightLeftLeft.getData());
        assertEquals(0, rightLeftLeft.getHeight());
        assertEquals(0, rightLeftLeft.getBalanceFactor());

        AVLNode<Integer> rightRight = right.getRight();
        assertEquals((Integer) 50, rightRight.getData());
        assertEquals(1, rightRight.getHeight());
        assertEquals(0, rightRight.getBalanceFactor());

        AVLNode<Integer> rightRightLeft = rightRight.getLeft();
        assertEquals((Integer) 48, rightRightLeft.getData());
        assertEquals(0, rightRightLeft.getHeight());
        assertEquals(0, rightRightLeft.getBalanceFactor());

        AVLNode<Integer> rightRightRight = rightRight.getRight();
        assertEquals((Integer) 55, rightRightRight.getData());
        assertEquals(0, rightRightRight.getHeight());
        assertEquals(0, rightRightRight.getBalanceFactor());

    }

    @Test(timeout = TIMEOUT)
    public void testAddMultipleRotations2() {
        tree.add(50);
        tree.add(25);
        tree.add(75);
        tree.add(20);
        tree.add(30);
        tree.add(70);
        tree.add(80);
        tree.add(10);
        tree.add(22);
        tree.add(27);
        tree.add(33);
        tree.add(65);
        tree.add(72);
        tree.add(85);
        tree.add(90); // left rotation
        tree.add(31);
        tree.add(32); // left-right rotation
        tree.add(64);
        tree.add(63); // right rotation
        tree.add(67); // left-right rotation
        tree.add(84);
        tree.add(82); // right-left rotation
        tree.add(29);
        tree.add(28);

        assertSame((Integer) 50, tree.get(50));
        assertSame((Integer) 30, tree.get(30));
        assertSame((Integer) 64, tree.get(64));
        assertSame((Integer) 82, tree.get(82));
        assertSame((Integer) 28, tree.get(28));
        assertSame((Integer) 90, tree.get(90));

        assertEquals(24, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 25, 20, 10, 22, 30, 28, 27, 29, 32, 31, 33, 75, 65, 64, 63, 70, 67, 72, 85, 82,
                80, 84, 90};
        Integer[] expectedBFS = {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
        Integer[] expectedHeights = {4, 3, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 3, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAddMultipleRotations3() {
        tree.add(49);
        tree.add(24);
        tree.add(10);
        tree.add(5);
        tree.add(9);
        tree.add(75);
        tree.add(100);
        tree.add(62);
        tree.add(54);

        assertEquals(9, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {24, 9, 5, 10, 75, 54, 49, 62, 100};
        Integer[] expectedBFS = {-1, 0, 0, 0, 1, 0, 0, 0, 0};
        Integer[] expectedHeights = {3, 1, 0, 0, 2, 1, 0, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testAddReverseOrder() {

        for (int i = 30; i >= 0; i--) {
            tree.add(i);
        }

        assertEquals(31, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {15, 7, 3, 1, 0, 2, 5, 4, 6, 11, 9, 8, 10, 13, 12, 14, 23, 19, 17, 16, 18, 21, 20,
                22, 27, 25, 24, 26, 29, 28, 30};
        Integer[] expectedBFS = new Integer[31];
        Integer[] expectedHeights = {4, 3, 2, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 3, 2, 1, 0, 0, 1, 0, 0, 2, 1, 0,
                0, 1, 0, 0};

        for (int i = 0; i <= 30; i++) {
            expectedBFS[i] = 0;
        }

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testAddDuplicates() {
        tree.add(27);
        tree.add(34);
        tree.add(43);
        tree.add(12);
        tree.add(18);
        tree.add(34);
        tree.add(27);
        tree.add(90);
        tree.add(12);

        assertEquals(6, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {34, 18, 12, 27, 43, 90};
        Integer[] expectedBFS = {0, 0, 0, 0, -1, 0};
        Integer[] expectedHeights = {2, 1, 0, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void insertAscendingOrder() {
        for (int i = 0; i <= 50; i += 2) {
            tree.add(i);
        }

        assertEquals(26, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {30, 14, 6, 2, 0, 4, 10, 8, 12, 22, 18, 16, 20, 26, 24, 28, 38, 34, 32, 36, 46, 42,
                40, 44, 48, 50};
        Integer[] expectedBFS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0};
        Integer[] expectedHeights = {4, 3, 2, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 3, 1, 0, 0, 2, 1, 0, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void insertDescendingOrder() {
        for (int i = 50; i >= 0; i -= 2) {
            tree.add(i);
        }

        assertEquals(26, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {20, 12, 4, 2, 0, 8, 6, 10, 16, 14, 18, 36, 28, 24, 22, 26, 32, 30, 34, 44, 40, 38,
                42, 48, 46, 50};
        Integer[] expectedBFS = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        Integer[] expectedHeights = {4, 3, 2, 1, 0, 1, 0, 0, 1, 0, 0, 3, 2, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOne() {
        tree.add(5);
        assertSame((Integer) 5, tree.remove(5));
        assertEquals(0, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertNull(root);

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOneFromTwo() {
        tree.add(5);
        tree.add(10);

        assertEquals(2, tree.size());

        assertSame((Integer) 5, tree.remove(5));
        assertEquals(1, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertEquals((Integer) 10, root.getData());
        assertEquals(0, root.getHeight());
        assertEquals(0, root.getBalanceFactor());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTwo() {
        tree.add(5);
        tree.add(10);

        assertEquals(2, tree.size());

        assertSame((Integer) 10, tree.remove(10));
        assertSame((Integer) 5, tree.remove(5));
        assertEquals(0, tree.size());

        AVLNode<Integer> root = tree.getRoot();
        assertNull(root);
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveEachRotation() {
        tree.add(50);
        tree.add(25);
        tree.add(75);
        tree.add(20);
        tree.add(30);
        tree.add(70);
        tree.add(80);
        tree.add(10);
        tree.add(22);
        tree.add(27);
        tree.add(33);
        tree.add(65);
        tree.add(72);
        tree.add(85);
        tree.add(90);
        tree.add(31);
        tree.add(32);
        tree.add(64);
        tree.add(63);
        tree.add(67);
        tree.add(84);
        tree.add(82);
        tree.add(29);
        tree.add(28);

        assertEquals(24, tree.size());
        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 25, 20, 10, 22, 30, 28, 27, 29, 32, 31, 33, 75, 65, 64, 63, 70, 67, 72, 85, 82,
                80, 84, 90};
        Integer[] expectedBFS = {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
        Integer[] expectedHeights = {4, 3, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 3, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

        assertSame((Integer) 65, tree.remove(65)); // no rotation needed
        assertEquals(23, tree.size());

        tree.add(26); // right-left rotation
        assertEquals(24, tree.size());

        assertSame((Integer) 31, tree.remove(31)); // no rotation needed
        assertEquals(23, tree.size());

        assertSame((Integer) 90, tree.remove(90)); // right rotation needed
        assertEquals(22, tree.size());

        assertSame((Integer) 80, tree.remove(80)); // right-left rotation needed
        assertEquals(21, tree.size());

        assertSame((Integer) 64, tree.remove(64)); // left rotation needed
        assertEquals(20, tree.size());

        assertSame((Integer) 82, tree.remove(82)); // no rotation needed
        assertEquals(19, tree.size());

        assertSame((Integer) 72, tree.remove(72)); // left-right rotation needed
        assertEquals(18, tree.size());

        assertSame((Integer) 33, tree.remove(33)); // no rotation needed
        assertEquals(17, tree.size());

        assertSame((Integer) 29, tree.remove(29)); // no rotation needed
        assertEquals(16, tree.size());

        assertSame((Integer) 50, tree.remove(50)); // right rotation needed
        assertEquals(15, tree.size());

        assertSame((Integer) 67, tree.remove(67)); // no rotation needed
        assertEquals(14, tree.size());

        assertSame((Integer) 84, tree.remove(84)); // no rotation needed
        assertEquals(13, tree.size());

        assertSame((Integer) 85, tree.remove(85)); // two left-right rotations needed
        assertEquals(12, tree.size());

        assertSame((Integer) 20, tree.remove(20)); // no rotations needed
        assertSame((Integer) 10, tree.remove(10)); // no rotations needed
        assertSame((Integer) 22, tree.remove(22)); // right-left rotation needed
        assertSame((Integer) 27, tree.remove(27)); // no rotations needed
        assertSame((Integer) 30, tree.remove(30)); // left rotation needed
        assertSame((Integer) 28, tree.remove(28)); // right-left rotation needed
        assertSame((Integer) 25, tree.remove(25)); // no rotations needed
        assertSame((Integer) 26, tree.remove(26)); // left rotation needed
        assertSame((Integer) 75, tree.remove(75)); // left-right rotation needed

        assertEquals(3, tree.size());

        values = preorder("v");
        heights = preorder("h");
        bfs = preorder("b");

        Integer[] expectedValues2 = {63, 32, 70};
        Integer[] expectedBFS2 = {0, 0, 0};
        Integer[] expectedHeights2 = {1, 0, 0};

        assertArrayEquals(expectedValues2, values.toArray());
        assertArrayEquals(expectedHeights2, heights.toArray());
        assertArrayEquals(expectedBFS2, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testAddNegatives() {
        tree.add(-5);
        tree.add(-3);
        tree.add(-4);

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {-4, -5, -3};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodesDescendingOrder() {
        for (int i = 10; i <= 100; i += 10) {
            tree.add(i);
        }

        assertEquals(10, tree.size());

        assertSame((Integer) 100, tree.remove(100));
        assertSame((Integer) 90, tree.remove(90));
        assertSame((Integer) 80, tree.remove(80));
        assertSame((Integer) 70, tree.remove(70));
        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 50, tree.remove(50));
        assertSame((Integer) 10, tree.remove(10));

        assertTrue(tree.contains(20));
        assertTrue(tree.contains(30));
        assertTrue(tree.contains(40));
        assertFalse(tree.contains(90));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {30, 20, 40};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNoRotations() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 20, tree.remove(20));
        assertSame((Integer) 30, tree.remove(30));
        assertSame((Integer) 80, tree.remove(80));

        assertSame((Integer) 90, tree.get(90));
        assertSame((Integer) 40, tree.get(40));
        assertSame((Integer) 50, tree.get(50));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 40, 90};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTriggerLeft() {
        tree.add(10);
        tree.add(20);
        tree.add(30);
        tree.add(25);
        tree.add(40);

        assertEquals(5, tree.size());
        assertSame((Integer) 10, tree.remove(10));

        assertEquals(4, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {30, 20, 25, 40};
        Integer[] expectedBFS = {1, -1, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTriggerLeftRight() {
        tree.add(10);
        tree.add(20);
        tree.add(30);
        tree.add(40);
        tree.add(8);
        tree.add(9);

        assertEquals(6, tree.size());

        assertSame((Integer) 8, tree.remove(8));

        assertEquals(5, tree.size());

        assertSame((Integer) 40, tree.remove(40));

        assertEquals(4, tree.size());

        assertSame((Integer) 30, tree.remove(30));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {10, 9, 20};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTriggerRight() {
        tree.add(10);
        tree.add(20);
        tree.add(30);
        tree.add(25);
        tree.add(9);
        tree.add(12);

        assertEquals(6, tree.size());

        assertSame((Integer) 30, tree.remove(30));

        assertEquals(5, tree.size());

        assertSame((Integer) 25, tree.remove(25));

        assertEquals(4, tree.size());

        assertTrue(tree.contains(10));
        assertTrue(tree.contains(9));
        assertTrue(tree.contains(20));
        assertTrue(tree.contains(12));
        assertFalse(tree.contains(25));

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {10, 9, 20, 12};
        Integer[] expectedBFS = {-1, 0, 1, 0};
        Integer[] expectedHeights = {2, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveTriggerRightLeft() {
        tree.add(40);
        tree.add(10);
        tree.add(30);
        tree.add(90);
        tree.add(50);
        tree.add(100);

        assertEquals(6, tree.size());

        assertSame((Integer) 100, tree.remove(100));
        assertSame((Integer) 10, tree.remove(10));
        assertSame((Integer) 30, tree.remove(30));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 40, 90};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveEvenNodes() {
        tree.add(45);
        tree.add(32);
        tree.add(76);
        tree.add(12);
        tree.add(35);
        tree.add(67);
        tree.add(78);
        tree.add(8);
        tree.add(43);
        tree.add(66);

        assertEquals(10, tree.size());

        assertSame((Integer) 32, tree.remove(32));
        assertSame((Integer) 76, tree.remove(76));
        assertSame((Integer) 78, tree.remove(78));
        assertSame((Integer) 8, tree.remove(8));
        assertSame((Integer) 12, tree.remove(12));
        assertSame((Integer) 66, tree.remove(66));

        assertEquals(4, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {45, 35, 43, 67};
        Integer[] expectedBFS = {1, -1, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveOddNodes() {
        tree.add(45);
        tree.add(32);
        tree.add(76);
        tree.add(12);
        tree.add(35);
        tree.add(67);
        tree.add(78);
        tree.add(8);
        tree.add(43);
        tree.add(66);

        assertEquals(10, tree.size());

        assertSame((Integer) 35, tree.remove(35));
        assertSame((Integer) 67, tree.remove(67));
        assertSame((Integer) 45, tree.remove(45));
        assertSame((Integer) 43, tree.remove(43));

        assertEquals(6, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {32, 12, 8, 76, 66, 78};
        Integer[] expectedBFS = {0, 1, 0, 0, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testNoRotationAfterRemovePredecessor() {
        tree.add(20);
        tree.add(40);
        tree.add(30);
        tree.add(10);
        tree.add(15);

        assertEquals(5, tree.size());

        assertSame((Integer) 30, tree.remove(30));
        assertEquals(4, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {20, 15, 10, 40};
        Integer[] expectedBFS = {1, 1, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testWithRotationAfterRemovePredecessor() {
        tree.add(20);
        tree.add(40);
        tree.add(30);
        tree.add(10);
        tree.add(15);
        tree.add(21);

        assertEquals(6, tree.size());

        assertSame((Integer) 20, tree.remove(20));
        assertSame((Integer) 15, tree.remove(15));
        assertEquals(4, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {30, 10, 21, 40};
        Integer[] expectedBFS = {1, -1, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithNoChildNoRotations() {
        tree.add(40);
        tree.add(23);
        tree.add(85);
        tree.add(25);
        tree.add(47);
        tree.add(89);

        assertEquals(6, tree.size());
        assertSame((Integer) 89, tree.remove(89));
        assertEquals(5, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {40, 23, 25, 85, 47};
        Integer[] expectedBFS = {0, -1, 0, 1, 0};
        Integer[] expectedHeights = {2, 1, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithNoChildWithRotation() {
        tree.add(32);
        tree.add(30);
        tree.add(45);
        tree.add(43);
        tree.add(48);

        assertEquals(5, tree.size());
        assertSame((Integer) 30, tree.remove(30));
        assertEquals(4, tree.size());

        assertSame((Integer) 45, tree.get(45));
        assertSame((Integer) 43, tree.get(43));
        assertSame((Integer) 48, tree.get(48));

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {45, 32, 43, 48};
        Integer[] expectedBFS = {1, -1, 0, 0};
        Integer[] expectedHeights = {2, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithOneChildNoRotations() {
        tree.add(40);
        tree.add(23);
        tree.add(85);
        tree.add(25);
        tree.add(47);
        tree.add(89);

        assertEquals(6, tree.size());
        assertSame((Integer) 23, tree.remove(23));
        assertEquals(5, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {40, 25, 85, 47, 89};
        Integer[] expectedBFS = {-1, 0, 0, 0, 0};
        Integer[] expectedHeights = {2, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithOneChildWithRotation() {
        tree.add(16);
        tree.add(12);
        tree.add(19);
        tree.add(11);
        tree.add(14);
        tree.add(18);
        tree.add(30);
        tree.add(13);
        tree.add(15);
        tree.add(25);
        tree.add(32);
        tree.add(17);
        tree.add(35);

        assertEquals(13, tree.size());
        assertSame((Integer) 18, tree.remove(18));
        assertEquals(12, tree.size());

        assertTrue(tree.contains(19));
        assertTrue(tree.contains(12));
        assertTrue(tree.contains(13));
        assertTrue(tree.contains(32));
        assertFalse(tree.contains(10));

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {16, 12, 11, 14, 13, 15, 30, 19, 17, 25, 32, 35};
        Integer[] expectedBFS = {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0};
        Integer[] expectedHeights = {3, 2, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithTwoChildrenNoRotations() {
        tree.add(40);
        tree.add(23);
        tree.add(85);
        tree.add(25);
        tree.add(47);
        tree.add(89);

        assertEquals(6, tree.size());
        assertSame((Integer) 85, tree.remove(85));
        assertEquals(5, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {40, 23, 25, 47, 89};
        Integer[] expectedBFS = {0, -1, 0, -1, 0};
        Integer[] expectedHeights = {2, 1, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveNodeWithTwoChildrenWithRotations() {
        tree.add(16);
        tree.add(12);
        tree.add(19);
        tree.add(11);
        tree.add(14);
        tree.add(18);
        tree.add(30);
        tree.add(13);
        tree.add(15);
        tree.add(25);
        tree.add(32);
        tree.add(17);
        tree.add(35);

        assertEquals(13, tree.size());

        assertSame((Integer) 16, tree.remove(16));
        assertSame((Integer) 19, tree.remove(19));
        assertSame((Integer) 30, tree.remove(30));
        assertSame((Integer) 12, tree.remove(12));

        assertEquals(9, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {15, 13, 11, 14, 25, 18, 17, 32, 35};
        Integer[] expectedBFS = {-1, 0, 0, 0, 0, 1, 0, -1, 0};
        Integer[] expectedHeights = {3, 1, 0, 0, 2, 1, 0, 1, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRoot() {
        tree.add(40);
        tree.add(23);
        tree.add(85);
        tree.add(25);
        tree.add(47);
        tree.add(89);

        assertEquals(6, tree.size());
        assertSame((Integer) 40, tree.remove(40));
        assertEquals(5, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {25, 23, 85, 47, 89};
        Integer[] expectedBFS = {-1, 0, 0, 0, 0};
        Integer[] expectedHeights = {2, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveAllLeaves1() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertEquals(7, tree.size());

        assertSame((Integer) 20, tree.remove(20));
        assertSame((Integer) 40, tree.remove(40));
        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 90, tree.remove(90));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 30, 80};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveAllLeaves2() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertEquals(7, tree.size());

        assertSame((Integer) 40, tree.remove(40));
        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 90, tree.remove(90));
        assertSame((Integer) 20, tree.remove(20));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 30, 80};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveAllLeaves3() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertEquals(7, tree.size());

        assertSame((Integer) 40, tree.remove(40));
        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 20, tree.remove(20));
        assertSame((Integer) 90, tree.remove(90));

        assertSame((Integer) 80, tree.get(80));
        assertSame((Integer) 30, tree.get(30));
        assertSame((Integer) 50, tree.get(50));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 30, 80};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveAllLeaves4() {
        tree.add(50);
        tree.add(30);
        tree.add(80);
        tree.add(20);
        tree.add(60);
        tree.add(90);
        tree.add(40);

        assertEquals(7, tree.size());

        assertSame((Integer) 60, tree.remove(60));
        assertSame((Integer) 40, tree.remove(40));
        assertSame((Integer) 20, tree.remove(20));
        assertSame((Integer) 90, tree.remove(90));

        assertEquals(3, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {50, 30, 80};
        Integer[] expectedBFS = {0, 0, 0};
        Integer[] expectedHeights = {1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRotationCascade1() {
        tree.add(21);
        tree.add(13);
        tree.add(29);
        tree.add(8);
        tree.add(18);
        tree.add(26);
        tree.add(32);
        tree.add(5);
        tree.add(11);
        tree.add(16);
        tree.add(20);
        tree.add(24);
        tree.add(28);
        tree.add(31);
        tree.add(33);
        tree.add(3);
        tree.add(7);
        tree.add(10);
        tree.add(12);
        tree.add(15);
        tree.add(17);
        tree.add(19);
        tree.add(23);
        tree.add(25);
        tree.add(27);
        tree.add(30);
        tree.add(2);
        tree.add(4);
        tree.add(6);
        tree.add(9);
        tree.add(14);
        tree.add(22);
        tree.add(1);

        assertEquals(33, tree.size());
        assertSame((Integer) 32, tree.remove(32)); // left rotation, then right rotation
        assertEquals(32, tree.size());

        assertTrue(tree.contains(3));
        assertTrue(tree.contains(4));
        assertTrue(tree.contains(21));
        assertTrue(tree.contains(19));
        assertFalse(tree.contains(232));

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {13, 8, 5, 3, 2, 1, 4, 7, 6, 11, 10, 9, 12, 21, 18, 16, 15, 14, 17, 20, 19, 26, 24,
                23, 22, 25, 29, 28, 27, 31, 30, 33};
        Integer[] expectedBFS = {0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                0, 0, 0};
        Integer[] expectedHeights = {5, 4, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 0, 4, 3, 2, 1, 0, 0, 1, 0, 3, 2, 1, 0, 0, 2,
                1, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testRemoveRotationCascade2() {
        Integer[] nums = {1000, 900, 2000, 800, 950, 1900, 3000, 700, 801, 930, 990, 1800, 1950, 2600, 4000, 802, 931,
                989, 995, 1700, 1830, 1920, 1970, 2200, 2800, 3300, 5000, 996, 1600, 1780, 1820, 1850, 1910, 1930, 1960,
                1980, 2100, 2300, 2700, 2900, 3100, 3500, 4800, 6000, 1500, 1601, 1779, 1781, 1819, 1821, 1849, 1851,
                1909, 1911, 1929, 1931, 1959, 1961, 1979, 1981, 2099, 2101, 2299, 2301, 2699, 2701, 2899, 2901, 3099,
                3101, 3499, 3501, 4799, 4801, 5999, 7000};

        for (int i = 0; i < nums.length; i++) {
            tree.add(nums[i]);
        }

        assertEquals(76, tree.size());

        Integer[] removes = {700, 990, 931, 930, 1000, 996, 1830, 3500, 4000, 3501, 3099, 3499, 5000, 7000, 6000, 2300,
                2100, 3300, 2600, 2299, 2101, 2800, 2901, 2701, 2699, 995, 950, 1700, 1851, 1780, 1820, 1819, 1849,
                1821, 1600, 1500, 2301, 2099, 2900, 2200, 3101, 4799, 3100, 800, 1900, 2899, 2000, 2700, 1981, 1980,
                1959, 1979, 1950, 1931, 1929, 1850, 1779, 1960, 1930, 3000, 1970, 4801};

        for (int j = 0; j < removes.length; j++) {
            tree.remove(removes[j]);
        }

        assertEquals((Integer) 1800, tree.get(1800));
        assertEquals((Integer) 989, tree.get(989));
        assertEquals((Integer) 802, tree.get(802));
        assertEquals((Integer) 801, tree.get(801));
        assertEquals((Integer) 1909, tree.get(1909));
        assertEquals((Integer) 5999, tree.get(5999));

        assertTrue(tree.contains(802));
        assertTrue(tree.contains(1601));
        assertTrue(tree.contains(801));
        assertTrue(tree.contains(1910));
        assertFalse(tree.contains(1332));

        assertEquals(14, tree.size());

        ArrayList<Integer> values = preorder("v");
        ArrayList<Integer> heights = preorder("h");
        ArrayList<Integer> bfs = preorder("b");

        Integer[] expectedValues = {1800, 989, 802, 801, 900, 1601, 1781, 1920, 1910, 1909, 1911, 4800, 1961, 5999};
        Integer[] expectedBFS = {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        Integer[] expectedHeights = {3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 0, 1, 0, 0};

        assertArrayEquals(expectedValues, values.toArray());
        assertArrayEquals(expectedHeights, heights.toArray());
        assertArrayEquals(expectedBFS, bfs.toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted1() {
        // full and complete tree

        Integer[] num = {20, 10, 40, 8, 11, 30, 50};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(8);
        deepest.add(11);
        deepest.add(40);
        deepest.add(30);
        deepest.add(50);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();
        sorted.add(8);
        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(7, 51).toArray());

        for (int i = 0; i < num.length; i++) {
            sorted.remove(0);
        }

        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(11, 30).toArray());

        sorted.remove(0);

        sorted.add(11);
        sorted.add(20);
        sorted.add(30);
        sorted.add(40);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(10, 50).toArray());

        for (int i = 0; i < 4; i++) {
            sorted.remove(0);
        }

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 10).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted2() {
        // leftside is deepest

        Integer[] num = {20, 10, 40, 7, 11, 30, 50, 6, 8};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(7);
        deepest.add(6);
        deepest.add(8);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(7);
        sorted.add(8);
        sorted.add(10);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(6, 11).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(30);
        sorted.add(40);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(20, 50).toArray());

        sorted.remove(0);
        sorted.remove(0);

        sorted.add(50);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(49, 51).toArray());

        sorted.remove(0);

        sorted.add(11);
        sorted.add(20);
        sorted.add(30);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(10, 40).toArray());

        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);

        sorted.add(8);
        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(7, 30).toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted3() {
        // leftside and rightside is deepest

        Integer[] num = {20, 10, 40, 7, 11, 30, 50, 6, 8, 51};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(7);
        deepest.add(6);
        deepest.add(8);
        deepest.add(40);
        deepest.add(50);
        deepest.add(51);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(7);
        sorted.add(8);
        sorted.add(10);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(6, 11).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(6);
        sorted.add(7);
        sorted.add(8);
        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(5, 52).toArray());

        for (int i = 0; i < 10; i++) {
            sorted.remove(0);
        }

        sorted.add(50);
        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(40, 52).toArray());

        sorted.remove(0);
        sorted.remove(0);

        sorted.add(11);
        sorted.add(20);
        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(10, 52).toArray());

        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);

        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(11, 30).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted4() {
        // leftside and rightside (multiple) is deepest

        Integer[] num = {20, 10, 40, 7, 11, 30, 50, 6, 8, 27, 51};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(7);
        deepest.add(6);
        deepest.add(8);
        deepest.add(40);
        deepest.add(30);
        deepest.add(27);
        deepest.add(50);
        deepest.add(51);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(27, 51).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(11);
        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(10, 21).toArray());

        for (int i = 0; i < 2; i++) {
            sorted.remove(0);
        }

        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 21).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(20);
        sorted.add(27);
        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(19, 52).toArray());

        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);
        sorted.remove(0);

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(28, 29).toArray());

        sorted.add(30);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(28, 31).toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted5() {
        // rightside is deepest

        Integer[] num = {20, 10, 40, 7, 11, 30, 51, 6, 8, 27, 50, 54, 55};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(40);
        deepest.add(51);
        deepest.add(54);
        deepest.add(55);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(51);
        sorted.add(54);
        sorted.add(55);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(50, 56).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(20);
        sorted.add(27);
        sorted.add(30);
        sorted.add(40);
        sorted.add(50);
        sorted.add(51);
        sorted.add(54);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(11, 55).toArray());

        for (int i = 0; i < 7; i++) {
            sorted.remove(0);
        }

        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 21).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(30);
        sorted.add(40);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(27, 50).toArray());

        sorted.remove(0);
        sorted.remove(0);

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(34, 38).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted6() {
        // rightside middle is deepest

        Integer[] num = {20, 10, 40, 7, 11, 30, 51, 6, 8, 27, 34, 50, 54, 26};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(40);
        deepest.add(30);
        deepest.add(27);
        deepest.add(26);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(8);
        sorted.add(10);
        sorted.add(11);
        sorted.add(20);
        sorted.add(26);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(7, 27).toArray());

        for (int i = 0; i < 5; i++) {
            sorted.remove(0);
        }

        sorted.add(34);
        sorted.add(40);
        sorted.add(50);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(30, 51).toArray());

        for (int i = 0; i < 3; i++) {
            sorted.remove(0);
        }

        sorted.add(6);
        sorted.add(7);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(-2, 8).toArray());

        for (int i = 0; i < 2; i++) {
            sorted.remove(0);
        }

        sorted.add(26);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(20, 27).toArray());

        sorted.remove(0);

        sorted.add(26);
        sorted.add(27);
        sorted.add(30);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(25, 31).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted7() {
        // both middle is deepest

        Integer[] num = {20, 10, 40, 7, 16, 30, 51, 6, 8, 11, 17, 27, 34, 50, 54, 18, 26};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(16);
        deepest.add(17);
        deepest.add(18);
        deepest.add(40);
        deepest.add(30);
        deepest.add(27);
        deepest.add(26);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(10);
        sorted.add(11);
        sorted.add(16);
        sorted.add(17);
        sorted.add(18);
        sorted.add(20);
        sorted.add(26);
        sorted.add(27);
        sorted.add(30);
        sorted.add(34);
        sorted.add(40);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(9, 41).toArray());

        for (int i = 0; i < 11; i++) {
            sorted.remove(0);
        }

        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(50, 54).toArray());

        for (int i = 0; i < 1; i++) {
            sorted.remove(0);
        }

        sorted.add(6);
        sorted.add(7);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(-2, 8).toArray());

        for (int i = 0; i < 2; i++) {
            sorted.remove(0);
        }

        sorted.add(7);
        sorted.add(8);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(6, 10).toArray());

        sorted.remove(0);
        sorted.remove(0);

        sorted.add(40);
        sorted.add(50);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(34, 51).toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted8() {
        // three middles is deepest

        Integer[] num = {20, 10, 40, 7, 16, 30, 51, 6, 8, 11, 17, 27, 34, 50, 54, 12, 18, 26};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(16);
        deepest.add(11);
        deepest.add(12);
        deepest.add(17);
        deepest.add(18);
        deepest.add(40);
        deepest.add(30);
        deepest.add(27);
        deepest.add(26);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(17);
        sorted.add(18);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(16, 19).toArray());

        for (int i = 0; i < 2; i++) {
            sorted.remove(0);
        }

        sorted.add(10);
        sorted.add(11);
        sorted.add(12);
        sorted.add(16);
        sorted.add(17);
        sorted.add(18);
        sorted.add(20);
        sorted.add(26);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(8, 27).toArray());

        for (int i = 0; i < 8; i++) {
            sorted.remove(0);
        }

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(27, 27).toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testDeepestBranchesAndSorted9() {
        // two leftsides is deepest

        Integer[] num = {20, 10, 40, 7, 16, 30, 51, 6, 8, 11, 17, 27, 34, 50, 54, 12, 18};

        for (int i = 0; i < num.length; i++) {
            tree.add(num[i]);
        }

        List<Integer> deepest = new ArrayList<>();
        deepest.add(20);
        deepest.add(10);
        deepest.add(16);
        deepest.add(11);
        deepest.add(12);
        deepest.add(17);
        deepest.add(18);
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());

        List<Integer> sorted = new ArrayList<>();

        sorted.add(30);
        sorted.add(34);
        sorted.add(40);
        sorted.add(50);
        sorted.add(51);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(27, 52).toArray());

        for (int i = 0; i < 5; i++) {
            sorted.remove(0);
        }

        sorted.add(11);
        sorted.add(12);
        sorted.add(16);
        sorted.add(17);
        sorted.add(18);
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(10, 20).toArray());

        for (int i = 0; i < 5; i++) {
            sorted.remove(0);
        }

        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(11, 12).toArray());

    }

    // one node deepest and sorted tested in addRoot()

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testConstructorAddNullData() {
        List<Integer> elements = null;
        tree = new AVL<>(elements);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testConstructorAddNullElements() {
        List<Integer> elements = new ArrayList<Integer>();
        elements.add(4);
        elements.add(null);
        tree = new AVL<>(elements);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testAddNullData() {
        tree.add(2);
        tree.add(null);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRemoveNullData() {
        tree.add(4);
        tree.add(23);
        tree.remove(null);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testRemoveNonExistentData() {
        tree.add(4);
        tree.add(23);
        tree.remove(7);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testGetNullData() {
        tree.add(4);
        tree.add(5);
        tree.get(null);
    }

    @Test(timeout = TIMEOUT)
    public void testGetNullRootHeight() {
        assertEquals(-1, tree.height());
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetNonExistentData() {
        tree.add(4);
        tree.add(23);
        tree.get(7);
    }

    @Test(timeout = TIMEOUT, expected = NoSuchElementException.class)
    public void testGetNonExistentDataAfterRemove() {
        tree.add(4);
        tree.add(23);
        tree.remove(23);
        tree.get(23);
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testContainsNullData() {
        tree.add(5);
        tree.contains(null);
    }

    @Test(timeout = TIMEOUT)
    public void testContainsExistentData() {
        tree.add(4);
        tree.add(23);
        assertTrue(tree.contains(4));
    }

    @Test(timeout = TIMEOUT)
    public void testContainsNonExistentData() {
        tree.add(4);
        tree.add(23);
        assertFalse(tree.contains(8));
    }

    @Test(timeout = TIMEOUT)
    public void testContainsNonExistentDataAfterRemove() {
        tree.add(4);
        tree.add(23);
        tree.remove(23);
        assertFalse(tree.contains(23));
    }

    @Test(timeout = TIMEOUT)
    public void testSingleNodeRootHeight() {
        tree.add(4);
        tree.add(23);
        assertEquals(1, tree.height());
    }

    @Test(timeout = TIMEOUT)
    public void testClear() {
        for (int i = 0; i <= 40; i++) {
            tree.add(i);
        }

        tree.clear();
        assertNull(tree.getRoot());
        assertEquals(0, tree.size());
    }

    @Test(timeout = TIMEOUT)
    public void testDeepestNullRoot() {
        List<Integer> deepest = new ArrayList<Integer>();
        assertArrayEquals(deepest.toArray(), tree.deepestBranches().toArray());
    }

    @Test(timeout = TIMEOUT)
    public void testSortedSameValues() {
        tree.add(4);
        tree.add(3);
        tree.add(5);

        List<Integer> sorted = new ArrayList<Integer>();
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(4, 4).toArray());

    }

    @Test(timeout = TIMEOUT)
    public void testSortedRangeNotWithinTree() {
        tree.add(4);
        tree.add(3);
        tree.add(5);

        List<Integer> sorted = new ArrayList<Integer>();
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(3, 4).toArray());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedNullInputs1() {
        tree.add(4);
        tree.add(3);
        tree.add(5);

        List<Integer> sorted = new ArrayList<Integer>();
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(null, 4).toArray());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedNullInputs2() {
        tree.add(4);
        tree.add(3);
        tree.add(5);

        List<Integer> sorted = new ArrayList<Integer>();
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(null, null).toArray());
    }

    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testSortedInvalidInputs() {
        tree.add(4);
        tree.add(3);
        tree.add(5);

        List<Integer> sorted = new ArrayList<Integer>();
        assertArrayEquals(sorted.toArray(), tree.sortedInBetween(5, 4).toArray());
    }

    private ArrayList<Integer> preorder(String letter) {
        ArrayList<Integer> values = new ArrayList<Integer>();
        ArrayList<Integer> heights = new ArrayList<Integer>();
        ArrayList<Integer> bfs = new ArrayList<Integer>();
        rpreorder(values, heights, bfs, tree.getRoot());

        if (letter.equals("v")) {
            return values;
        } else if (letter.equals("h")) {
            return heights;
        }

        return bfs;
    }

    private void rpreorder(ArrayList<Integer> values, ArrayList<Integer> heights, ArrayList<Integer> bfs,
                           AVLNode<Integer> curr) {
        if (curr == null) {
            return;
        }

        values.add(curr.getData());
        heights.add(curr.getHeight());
        bfs.add(curr.getBalanceFactor());
        rpreorder(values, heights, bfs, curr.getLeft());
        rpreorder(values, heights, bfs, curr.getRight());
    }

}
