import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * So far, this just includes everything gone over in lecture.
 *
 * @author Riley Corzine
 * @version 1.0
 */
public class ExtraPatternMatchingTest {

    private static final int TIMEOUT = 200;

    @Test
    public void testBuildFailureTableWithSingleCharPattern() {
        String pattern = "a";
        CharacterComparator comparator = new CharacterComparator();
        int[] expected = {0};
        int[] result = PatternMatching.buildFailureTable(pattern, comparator);
        assertArrayEquals(expected, result);
    }

    @Test
    public void testBuildFailureTableWithEmptyPattern() {
        String pattern = "";
        CharacterComparator comparator = new CharacterComparator();
        int[] expected = {};
        int[] result = PatternMatching.buildFailureTable(pattern, comparator);
        assertArrayEquals(expected, result);
    }
    @Test(expected = IllegalArgumentException.class)
    public void testBuildFailureTableWithNullPattern() {
        String pattern = null;
        CharacterComparator comparator = new CharacterComparator();
        int[] result = PatternMatching.buildFailureTable(pattern, comparator);
    }
    @Test(expected = IllegalArgumentException.class)
    public void testBuildFailureTableWithNullComparator() {
        String pattern = "abcabc";
        CharacterComparator comparator = null;
        int[] result = PatternMatching.buildFailureTable(pattern, comparator);
    }

    @Test
    public void testBoyerMooreValidInput() {
        CharSequence pattern = "abc";
        CharSequence text = "ababcabcababccbacba";
        CharacterComparator comparator = new CharacterComparator();
        List<Integer> expectedMatches = new ArrayList<>();
        expectedMatches.add(2);
        expectedMatches.add(5);
        expectedMatches.add(10);
        assertEquals(expectedMatches, PatternMatching.boyerMoore(pattern, text, comparator));
    }
    @Test
    public void testNoMatch() {
        CharSequence pattern = "xyz";
        CharSequence text = "abcdefghi";
        CharacterComparator comparator = new CharacterComparator();
        List<Integer> expectedMatches = new ArrayList<>();
        assertEquals(expectedMatches, PatternMatching.boyerMoore(pattern, text, comparator));
    }
    @Test
    public void testNullPattern() {
        CharSequence pattern = null;
        CharSequence text = "abcdefghi";
        CharacterComparator comparator = new CharacterComparator();
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(pattern, text, comparator));
    }
    @Test
    public void testEmptyPattern() {
        CharSequence pattern = "";
        CharSequence text = "abcdefghi";
        CharacterComparator comparator = new CharacterComparator();
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(pattern, text, comparator));
    }
    @Test
    public void testNullText() {
        CharSequence pattern = "abc";
        CharSequence text = null;
        CharacterComparator comparator = new CharacterComparator();
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(pattern, text, comparator));
    }
    @Test
    public void testNullComparator() {
        CharSequence pattern = "abc";
        CharSequence text = "ababcabcababccbacba";
        CharacterComparator comparator = null;
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(pattern, text, comparator));
    }
}