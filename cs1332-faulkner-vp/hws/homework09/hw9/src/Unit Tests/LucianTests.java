import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Pattern Matching Algorithm JUnits
 *
 * There are 50-100 tests for each method using randomly generated text & patterns.
 * Expected comparison counts are from https://csvistool.com/. It might be helpful to use this for debugging.
 *
 * @author Lucian Tash
 * @version 1.0
 */
public class LucianTests {

    private static final int TIMEOUT = 200;
    private CharacterComparator comparator;
    private String patternMatch;
    private String patternNoMatch;
    private String text;
    private ArrayList<Integer> matches;
    private ArrayList<Integer> empty;
    private static boolean skipGalil = false;

    @Before
    public void setUp() {
        comparator = new CharacterComparator();
        patternMatch = "";
        patternNoMatch = "";
        text = "";
        matches = new ArrayList<>();
        empty = new ArrayList<>();
    }

    @BeforeClass
    public static void galilWarn() {
        if (PatternMatching.boyerMooreGalilRule("a", "a", new CharacterComparator()) == null) {
            System.err.println("WARNING: Automatically passing tests for Galil Rule because you did not implement it.");
            skipGalil = true;
        }
    }

    @Test(timeout = TIMEOUT)
    public void testKMPExceptions() {
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.kmp("", "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.kmp(null, "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.kmp(null, null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.kmp("pattern!", null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.kmp("pattern!", "text!", null));
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreExceptions() {
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore("", "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(null, "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore(null, null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore("pattern!", null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMoore("pattern!", "text!", null));
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpExceptions() {
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.rabinKarp("", "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.rabinKarp(null, "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.rabinKarp(null, null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.rabinKarp("pattern!", null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.rabinKarp("pattern!", "text!", null));
    }

    @Test(timeout = TIMEOUT)
    public void testGalilExceptions() {
        if (skipGalil) return;
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMooreGalilRule("", "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMooreGalilRule(null, "text!", comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMooreGalilRule(null, null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMooreGalilRule("pattern!", null, comparator));
        assertThrows(IllegalArgumentException.class, () -> PatternMatching.boyerMooreGalilRule("pattern!", "text!", null));
    }

    @Test(timeout = TIMEOUT)
    public void testKMPPatternTooLarge() {
        text = "word";
        patternNoMatch = "word!";
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoorePatternTooLarge() {
        text = "word";
        patternNoMatch = "word!";
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpPatternTooLarge() {
        text = "word";
        patternNoMatch = "word!";
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilPatternTooLarge() {
        if (skipGalil) return;
        text = "word";
        patternNoMatch = "word!";
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilEmptyText() {
        if (skipGalil) return;
        text = "";
        patternNoMatch = "word";
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMPEmptyText() {
        text = "";
        patternNoMatch = "word";
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreEmptyText() {
        text = "";
        patternNoMatch = "word";
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpEmptyText() {
        text = "";
        patternNoMatch = "word";
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilExactMatch() {
        if (skipGalil) return;
        text = "word.";
        patternNoMatch = "word.";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMPExactMatch() {
        text = "word.";
        patternNoMatch = "word.";
        matches.add(0);
        assertEquals(matches, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreExactMatch() {
        text = "word.";
        patternNoMatch = "word.";
        matches.add(0);
        assertEquals(matches, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpExactMatch() {
        text = "word.";
        patternNoMatch = "word.";
        matches.add(0);
        assertEquals(matches, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalilManyMatches() {
        if (skipGalil) return;
        text = "hihihihihihihihihihi";
        patternMatch = "hih";
        matches.add(0);
        matches.add(2);
        matches.add(4);
        matches.add(6);
        matches.add(8);
        matches.add(10);
        matches.add(12);
        matches.add(14);
        matches.add(16);
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMPManyMatches() {
        text = "hihihihihihihihihihi";
        patternMatch = "hih";
        matches.add(0);
        matches.add(2);
        matches.add(4);
        matches.add(6);
        matches.add(8);
        matches.add(10);
        matches.add(12);
        matches.add(14);
        matches.add(16);
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMooreManyMatches() {
        text = "hihihihihihihihihihi";
        patternMatch = "hih";
        matches.add(0);
        matches.add(2);
        matches.add(4);
        matches.add(6);
        matches.add(8);
        matches.add(10);
        matches.add(12);
        matches.add(14);
        matches.add(16);
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarpManyMatches() {
        text = "hihihihihihihihihihi";
        patternMatch = "hih";
        matches.add(0);
        matches.add(2);
        matches.add(4);
        matches.add(6);
        matches.add(8);
        matches.add(10);
        matches.add(12);
        matches.add(14);
        matches.add(16);
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_1() {
        patternMatch = "aaacdcdbdcbad";
        int[] table = {0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_2() {
        patternMatch = "ddbdabccbcaabdcaacabaab";
        int[] table = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_3() {
        patternMatch = "abdcccacddbbaabacdddcbbcdbcbb";
        int[] table = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_4() {
        patternMatch = "abaadaaddcdcbdcadcbdadc";
        int[] table = {0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_5() {
        patternMatch = "daddaccaccbdd";
        int[] table = {0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_6() {
        patternMatch = "ccbcbbdcaacacdabcb";
        int[] table = {0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_7() {
        patternMatch = "badbcbccacbddbaabdcdc";
        int[] table = {0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_8() {
        patternMatch = "dbddbbbcbbdbbadcbdccdaddcd";
        int[] table = {0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_9() {
        patternMatch = "b";
        int[] table = {0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_10() {
        patternMatch = "aaddbdabbabbabbadcbcbdccaadcbd";
        int[] table = {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_11() {
        patternMatch = "bbcabaabd";
        int[] table = {0, 1, 0, 0, 1, 0, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_12() {
        patternMatch = "dcbcaccaadadadbddbb";
        int[] table = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_13() {
        patternMatch = "ddcbabbdadbbacbdabd";
        int[] table = {0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_14() {
        patternMatch = "cdbcdaabadaccdadcbbcbaddbbc";
        int[] table = {0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 31, was " + comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_15() {
        patternMatch = "adbcdcddcbbddcbbcbd";
        int[] table = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_16() {
        patternMatch = "acbbcbaaccddc";
        int[] table = {0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_17() {
        patternMatch = "caaba";
        int[] table = {0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_18() {
        patternMatch = "ccacabcadbaadaacbaccabcb";
        int[] table = {0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 3, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_19() {
        patternMatch = "bdcdcbadbcccada";
        int[] table = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_20() {
        patternMatch = "baacdcadababcccbdd";
        int[] table = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 20, was " + comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_21() {
        patternMatch = "acbcdabdab";
        int[] table = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_22() {
        patternMatch = "abdcd";
        int[] table = {0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_23() {
        patternMatch = "aaa";
        int[] table = {0, 1, 2};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_24() {
        patternMatch = "cabdadbcca";
        int[] table = {0, 0, 0, 0, 0, 0, 0, 1, 1, 2};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_25() {
        patternMatch = "bbddcdbadcabacaccddbddacbcdca";
        int[] table = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_26() {
        patternMatch = "cadccdcddddcbcdccb";
        int[] table = {0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 24, was " + comparator.getComparisonCount(), 24 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_27() {
        patternMatch = "badcaabdcaacbdacc";
        int[] table = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_28() {
        patternMatch = "b";
        int[] table = {0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_29() {
        patternMatch = "ccdbdccccaddddaadd";
        int[] table = {0, 1, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_30() {
        patternMatch = "abaddadabcabccdabbbadadbc";
        int[] table = {0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 31, was " + comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_31() {
        patternMatch = "dcaa";
        int[] table = {0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_32() {
        patternMatch = "adddabdccabbd";
        int[] table = {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_33() {
        patternMatch = "bcdbcbbaacdadbddbab";
        int[] table = {0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_34() {
        patternMatch = "dcaddbbdbddb";
        int[] table = {0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_35() {
        patternMatch = "bdcaa";
        int[] table = {0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_36() {
        patternMatch = "bcddabcabdcdcaaacdaadcacadc";
        int[] table = {0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_37() {
        patternMatch = "abddaddadbdcccddbccabadba";
        int[] table = {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_38() {
        patternMatch = "dcabbcbcbacccbaddbcadabddaabcb";
        int[] table = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 34, was " + comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_39() {
        patternMatch = "bdbcabdbbababddcccbab";
        int[] table = {0, 0, 1, 0, 0, 1, 2, 3, 1, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_40() {
        patternMatch = "abbb";
        int[] table = {0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_41() {
        patternMatch = "bbbbbbdaabdddbbaadcdbbdabcc";
        int[] table = {0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 37, was " + comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_42() {
        patternMatch = "bddbaacbbdacddad";
        int[] table = {0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_43() {
        patternMatch = "cacdcbbadadabdbdbccbbdcabdbdcb";
        int[] table = {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_44() {
        patternMatch = "bbbbdaddbdacadddbcccdcddabacb";
        int[] table = {0, 1, 2, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 34, was " + comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_45() {
        patternMatch = "bdac";
        int[] table = {0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_46() {
        patternMatch = "bada";
        int[] table = {0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_47() {
        patternMatch = "bdcbdbdbcac";
        int[] table = {0, 0, 0, 1, 2, 1, 2, 1, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 13, was " + comparator.getComparisonCount(), 13 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_48() {
        patternMatch = "ddbabcdcdccddbaccc";
        int[] table = {0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 3, 4, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_49() {
        patternMatch = "adcadbcbaaaddabbcbb";
        int[] table = {0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable_50() {
        patternMatch = "accbdaadb";
        int[] table = {0, 0, 0, 0, 0, 1, 1, 0, 0};
        assertArrayEquals(table, PatternMatching.buildFailureTable(patternMatch, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_1() {
        patternMatch = "bdaaadbabdbccbabaacddbacbcb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 26);
        table.put('c', 25);
        table.put('a', 22);
        table.put('d', 20);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_2() {
        patternMatch = "bcddaddabbbadbbaaaccbbabbdaa";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 27);
        table.put('d', 25);
        table.put('b', 24);
        table.put('c', 19);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_3() {
        patternMatch = "ddbdaadbacccddbcaabcdbacbad";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 26);
        table.put('a', 25);
        table.put('b', 24);
        table.put('c', 23);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_4() {
        patternMatch = "cacbaabbcadcbdccbda";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 18);
        table.put('d', 17);
        table.put('b', 16);
        table.put('c', 15);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_5() {
        patternMatch = "dddcaaadbaaccc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 13);
        table.put('a', 10);
        table.put('b', 8);
        table.put('d', 7);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_6() {
        patternMatch = "badbdaaabccdaadbbbaacaa";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 22);
        table.put('c', 20);
        table.put('b', 17);
        table.put('d', 14);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_7() {
        patternMatch = "baaccdcaaccaad";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 13);
        table.put('a', 12);
        table.put('c', 10);
        table.put('b', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_8() {
        patternMatch = "bdbaacaacacd";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 11);
        table.put('c', 10);
        table.put('a', 9);
        table.put('b', 2);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_9() {
        patternMatch = "dcaaadbbbdbbaaabbddacdc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 22);
        table.put('d', 21);
        table.put('a', 19);
        table.put('b', 16);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_10() {
        patternMatch = "daccdadbbacdaacbdbbaa";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 20);
        table.put('b', 18);
        table.put('d', 16);
        table.put('c', 14);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_11() {
        patternMatch = "bccaccbacccacaadcabbcaabccc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 26);
        table.put('b', 23);
        table.put('a', 22);
        table.put('d', 15);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_12() {
        patternMatch = "ddadadabbdbdbabbbaab";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 19);
        table.put('a', 18);
        table.put('d', 11);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_13() {
        patternMatch = "aadcdacdcddbabddb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 16);
        table.put('d', 15);
        table.put('a', 12);
        table.put('c', 8);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_14() {
        patternMatch = "bbcdabddbaaccbbdbaddcccdcaadb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 28);
        table.put('d', 27);
        table.put('a', 26);
        table.put('c', 24);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_15() {
        patternMatch = "aadbaadcdcaabcb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 14);
        table.put('c', 13);
        table.put('a', 11);
        table.put('d', 8);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_16() {
        patternMatch = "bdaacdcccdaad";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 12);
        table.put('a', 11);
        table.put('c', 8);
        table.put('b', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_17() {
        patternMatch = "dcacbcbabaadcc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 13);
        table.put('d', 11);
        table.put('a', 10);
        table.put('b', 8);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_18() {
        patternMatch = "adccababdcabaa";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 13);
        table.put('b', 11);
        table.put('c', 9);
        table.put('d', 8);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_19() {
        patternMatch = "dabcaabbdabdbbaddcbcc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 20);
        table.put('b', 18);
        table.put('d', 16);
        table.put('a', 14);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_20() {
        patternMatch = "aaccdabadbadaabaababdcb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 22);
        table.put('c', 21);
        table.put('d', 20);
        table.put('a', 18);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_21() {
        patternMatch = "ad";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 1);
        table.put('a', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_22() {
        patternMatch = "aacacadadadab";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 12);
        table.put('a', 11);
        table.put('d', 10);
        table.put('c', 4);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_23() {
        patternMatch = "ac";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 1);
        table.put('a', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_24() {
        patternMatch = "ddccccbdaadbbccabddacccacb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 25);
        table.put('c', 24);
        table.put('a', 23);
        table.put('d', 18);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_25() {
        patternMatch = "cc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 1);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_26() {
        patternMatch = "ddbccbdbaccacdbadacdcabcdab";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 26);
        table.put('a', 25);
        table.put('d', 24);
        table.put('c', 23);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_27() {
        patternMatch = "bbddbdcdcdacccdbbccbbc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 21);
        table.put('b', 20);
        table.put('d', 14);
        table.put('a', 10);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_28() {
        patternMatch = "dbdcdaabddbdabaadddb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 19);
        table.put('d', 18);
        table.put('a', 15);
        table.put('c', 3);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_29() {
        patternMatch = "dbdadcddbdbcaab";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 14);
        table.put('a', 13);
        table.put('c', 11);
        table.put('d', 9);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_30() {
        patternMatch = "dcccdb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 5);
        table.put('d', 4);
        table.put('c', 3);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_31() {
        patternMatch = "ccbabdabdbcacddcbaacdadbbb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 25);
        table.put('d', 22);
        table.put('a', 21);
        table.put('c', 19);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_32() {
        patternMatch = "cbbbb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 4);
        table.put('c', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_33() {
        patternMatch = "ccbabadaaadabcaaaabcdbabca";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 25);
        table.put('c', 24);
        table.put('b', 23);
        table.put('d', 20);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_34() {
        patternMatch = "bbdbcdbbabbbca";
        Map<Character, Integer> table = new HashMap<>();
        table.put('a', 13);
        table.put('c', 12);
        table.put('b', 11);
        table.put('d', 5);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_35() {
        patternMatch = "bdacacbcabdcbadccbbbadcdabcbb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 28);
        table.put('c', 26);
        table.put('a', 24);
        table.put('d', 23);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_36() {
        patternMatch = "bb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 1);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_37() {
        patternMatch = "bcabc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 4);
        table.put('b', 3);
        table.put('a', 2);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_38() {
        patternMatch = "bdbabbcbdccaabb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 14);
        table.put('a', 12);
        table.put('c', 10);
        table.put('d', 8);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_39() {
        patternMatch = "dcbcbaadcadbbbdcbb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 17);
        table.put('c', 15);
        table.put('d', 14);
        table.put('a', 9);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_40() {
        patternMatch = "bbbaccbadacacdc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 14);
        table.put('d', 13);
        table.put('a', 11);
        table.put('b', 6);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_41() {
        patternMatch = "cbbddacabacacb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 13);
        table.put('c', 12);
        table.put('a', 11);
        table.put('d', 4);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_42() {
        patternMatch = "bdcbdadbccbbbddadabc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 19);
        table.put('b', 18);
        table.put('a', 17);
        table.put('d', 16);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_43() {
        patternMatch = "adccdcacad";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 9);
        table.put('a', 8);
        table.put('c', 7);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_44() {
        patternMatch = "aaccccbcb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 8);
        table.put('c', 7);
        table.put('a', 1);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_45() {
        patternMatch = "bbbabd";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 5);
        table.put('b', 4);
        table.put('a', 3);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_46() {
        patternMatch = "adadbcdacdacbc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 13);
        table.put('b', 12);
        table.put('a', 10);
        table.put('d', 9);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_47() {
        patternMatch = "bd";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 1);
        table.put('b', 0);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_48() {
        patternMatch = "cccbaaabaababc";
        Map<Character, Integer> table = new HashMap<>();
        table.put('c', 13);
        table.put('b', 12);
        table.put('a', 11);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_49() {
        patternMatch = "bdaddbbbddabacadacbdd";
        Map<Character, Integer> table = new HashMap<>();
        table.put('d', 20);
        table.put('b', 18);
        table.put('c', 17);
        table.put('a', 16);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBuildLastTable_50() {
        patternMatch = "ccaccbbcabbacdcbabbadabcbcb";
        Map<Character, Integer> table = new HashMap<>();
        table.put('b', 26);
        table.put('c', 25);
        table.put('a', 21);
        table.put('d', 20);
        assertEquals(table, PatternMatching.buildLastTable(patternMatch));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_1() {
        text = "thing casewatertimegovernment-thing.partcaseprogramdaynight study.issue.lotchild";
        patternMatch = "thing.";
        patternNoMatch = "week";
        matches.add(30);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 88, was " + comparator.getComparisonCount(), 88 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 81, was " + comparator.getComparisonCount(), 81 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_2() {
        text = "storyweekword.way.school.programstoryareaschoolweek-homerightfamily weekcountryyearwaterstorylifelifewordlife.state.wordday roomareastatepointcountrycase-rightpartwordstatenightnightword area-place statefamilyissuetimesystemplace";
        patternMatch = "part";
        patternNoMatch = "money";
        matches.add(159);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 233, was " + comparator.getComparisonCount(), 233 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 236, was " + comparator.getComparisonCount(), 236 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_3() {
        text = "life system-issue-wordcompanymonth-studytime.year governmentgovernmentyearchildmonthareapeople.stateprogrammonthlotfamily programbook-roomsystemweek-schoolthingdaypeopleschoolchild statearea";
        patternMatch = "school";
        patternNoMatch = "student";
        matches.add(149);
        matches.add(169);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 199, was " + comparator.getComparisonCount(), 199 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 201, was " + comparator.getComparisonCount(), 201 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_4() {
        text = "group-pointcountry-program placelotplacegovernmentfamilyarea.child-familyprogram.moneystudy wayprogram.wayplace-thing schoolstudycountrygroupmonthyearnumberplace right.rightgroupfact";
        patternMatch = "group";
        patternNoMatch = "book";
        matches.add(0);
        matches.add(136);
        matches.add(173);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 189, was " + comparator.getComparisonCount(), 189 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 183, was " + comparator.getComparisonCount(), 183 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_5() {
        text = "familystudent.montharea.family-storyweekroom-waterfactarea system thingbook factstudent nightword study.timeweekcompanystate studyareastorytimeyearprogram rightmoneystudentstudy.study.month areaissuechildbook-";
        patternMatch = "study.";
        patternNoMatch = "lot";
        matches.add(98);
        matches.add(172);
        matches.add(178);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 220, was " + comparator.getComparisonCount(), 220 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 212, was " + comparator.getComparisonCount(), 212 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_6() {
        text = "schoolpeople casesystem-storystudentmonth.lot-governmentpointwaywaterworldstudylot-right room-way-partwordwater.programnumbermonthlotplacepointpointfamily.time case system";
        patternMatch = "word";
        patternNoMatch = "home";
        matches.add(102);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 176, was " + comparator.getComparisonCount(), 176 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 175, was " + comparator.getComparisonCount(), 175 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_7() {
        text = "numbermonth-factschool programpeople.studentpartthingyear-waterroomweekgroupsystem.yearweekstategovernment.way";
        patternMatch = "school ";
        patternNoMatch = "family";
        matches.add(16);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 114, was " + comparator.getComparisonCount(), 114 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 111, was " + comparator.getComparisonCount(), 111 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_8() {
        text = "area.lothomesystemwaypeopletime-world.company.rightfactpeoplenumbercompany.word-way ";
        patternMatch = "number";
        patternNoMatch = "school";
        matches.add(61);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 86, was " + comparator.getComparisonCount(), 86 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 86, was " + comparator.getComparisonCount(), 86 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_9() {
        text = "issuestudy thingissuestudywaterstudentyearschoolgroupissue issueworldmonth-moneywater.waterplacecompanycountrygroupmonthprogram";
        patternMatch = "issue";
        patternNoMatch = "night";
        matches.add(0);
        matches.add(16);
        matches.add(53);
        matches.add(59);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 128, was " + comparator.getComparisonCount(), 128 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 134, was " + comparator.getComparisonCount(), 134 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_10() {
        text = "areawayissuestate studentwordcountrystatecountryprogram.school.wayarea.companystudentprogramcountrygroup.week.rightprogram nightstate case.numbercasestoryweekissuelifeworldissueweekday booktime numbernight";
        patternMatch = "group.";
        patternNoMatch = "fact";
        matches.add(99);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 210, was " + comparator.getComparisonCount(), 210 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 206, was " + comparator.getComparisonCount(), 206 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_11() {
        text = "factbook countryfactstudentsystem.weekthingfactweek-studydaychildstudystorypointrighttimebookchildsystem-waygroup wordlife.moneyfactlothome-grouplotweekpoint systemworld-weekpartcaselife-part-pointpeople.";
        patternMatch = "lot";
        patternNoMatch = "area";
        matches.add(132);
        matches.add(145);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 210, was " + comparator.getComparisonCount(), 210 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 213, was " + comparator.getComparisonCount(), 213 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_12() {
        text = "weekwater partcompany fact roomdaygovernment part waycountrychildschoolstudymoneypointhome";
        patternMatch = "child";
        patternNoMatch = "people";
        matches.add(60);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 94, was " + comparator.getComparisonCount(), 94 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 95, was " + comparator.getComparisonCount(), 95 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_13() {
        text = "lotstudentprogramnightwaterwaypeoplepart schoolstoryareapoint-familylotyearyearcompanypartroom";
        patternMatch = "area";
        patternNoMatch = "world";
        matches.add(52);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 104, was " + comparator.getComparisonCount(), 104 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 96, was " + comparator.getComparisonCount(), 96 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_14() {
        text = "day.issueprogramstorychildtime";
        patternMatch = "day.";
        patternNoMatch = "system";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 31, was " + comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 34, was " + comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_15() {
        text = "night areastatechildtimelifeschool-lifebookwordschoolbook thingmoneystory.thingfact";
        patternMatch = "area";
        patternNoMatch = "day";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 85, was " + comparator.getComparisonCount(), 85 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 85, was " + comparator.getComparisonCount(), 85 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_16() {
        text = "yearnumber nightplacegroup-family.time-school-way.casesystem.childwatercompanyprogrammoneyschool world";
        patternMatch = "night";
        patternNoMatch = "life";
        matches.add(11);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 105, was " + comparator.getComparisonCount(), 105 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 107, was " + comparator.getComparisonCount(), 107 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_17() {
        text = "day-groupbook placestatefamily";
        patternMatch = "family";
        patternNoMatch = "month";
        matches.add(24);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_18() {
        text = "story ";
        patternMatch = "story ";
        patternNoMatch = "state";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_19() {
        text = "daycountrywordpeople-";
        patternMatch = "day";
        patternNoMatch = "way";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_20() {
        text = "story part-company.familymoneyrightcompany.home.countryschoolprogram-waybookcompanyissue moneyplaceplace.groupfamilytimewaterpartfamily-nightcasefamilyareaweeknumbersystemcompanyfamilylotpoint-groupstatestatemonthpoint-areapoint";
        patternMatch = "country";
        patternNoMatch = "child";
        matches.add(48);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 236, was " + comparator.getComparisonCount(), 236 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 237, was " + comparator.getComparisonCount(), 237 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_21() {
        text = "peopleweekpartcompanyrightissuebook";
        patternMatch = "issue";
        patternNoMatch = "group";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_22() {
        text = "peoplewaterrighthomefact world-right.";
        patternMatch = "fact ";
        patternNoMatch = "company";
        matches.add(20);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 37, was " + comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 38, was " + comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_23() {
        text = "water.programcompany.lifesystemcountrylifeareahomeright grouplife.schoolwater state day issueroomarea money weekcompanyfamilygrouptimelot.";
        patternMatch = "lot.";
        patternNoMatch = "child";
        matches.add(134);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 146, was " + comparator.getComparisonCount(), 146 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 142, was " + comparator.getComparisonCount(), 142 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_24() {
        text = "lotworldnumber child.areagovernmentbookbook waterlife.issueweek-nightcompanylotissuesystemtime companyfamily homepeoplestory bookpartway-area systemstoryyearmonthfactgovernmentcase.roomweekarea.schoolmoneywaterthingarea";
        patternMatch = "water";
        patternNoMatch = "program";
        matches.add(44);
        matches.add(205);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 223, was " + comparator.getComparisonCount(), 223 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 224, was " + comparator.getComparisonCount(), 224 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_25() {
        text = "casestudy.placenumber issue areastudentlifechild-childstorypeoplemonthcompanypoint.studylothomefacthomenightplace-worldcompany.daynumberfamilyworldmonth-placestudypartfactmoneystudent lotsystem";
        patternMatch = "company";
        patternNoMatch = "government";
        matches.add(70);
        matches.add(119);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 201, was " + comparator.getComparisonCount(), 201 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 194, was " + comparator.getComparisonCount(), 194 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_26() {
        text = "lifefamilyyear-weekbookschool-time-issuesystem story yearsystem.year-peoplegroupday-timeyearcasecountry-pointschool";
        patternMatch = "school";
        patternNoMatch = "room";
        matches.add(23);
        matches.add(109);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 128, was " + comparator.getComparisonCount(), 128 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 122, was " + comparator.getComparisonCount(), 122 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_27() {
        text = "water-year schoollotthingissueareaworldwaterdayarea.numberyear";
        patternMatch = "issue";
        patternNoMatch = "country";
        matches.add(25);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 63, was " + comparator.getComparisonCount(), 63 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 63, was " + comparator.getComparisonCount(), 63 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_28() {
        text = "partissue-weekpartbookcompanyfactwordhome.placefamilynight-lotprogram-state word.monthwaterprogramrightgrouppeoplehome casearea water-studentword waterstudenttime-companynightstudycompanystatenumber-case book-";
        patternMatch = "word ";
        patternNoMatch = "country";
        matches.add(141);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 215, was " + comparator.getComparisonCount(), 215 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 216, was " + comparator.getComparisonCount(), 216 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_29() {
        text = "lifelotmoneytimestudentissuemonthsystemgovernmentday peoplefactstudentcaseissuefact childlifelifethingworld state-worldpartbook";
        patternMatch = "case";
        patternNoMatch = "study";
        matches.add(70);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 130, was " + comparator.getComparisonCount(), 130 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 137, was " + comparator.getComparisonCount(), 137 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_30() {
        text = "thingweek.room.statelifeworld-pointword";
        patternMatch = "room.";
        patternNoMatch = "year";
        matches.add(10);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 40, was " + comparator.getComparisonCount(), 40 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 39, was " + comparator.getComparisonCount(), 39 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_31() {
        text = "money systemhomesystem-lotprogramfamilyarealifestatelotgovernmentroom childfactcountrystudywatercasenumbernightyeargovernment-lotcase yearsystem.night-grouplifeway.book";
        patternMatch = "state";
        patternNoMatch = "right";
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 177, was " + comparator.getComparisonCount(), 177 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 180, was " + comparator.getComparisonCount(), 180 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_32() {
        text = "timemonth-countrystory lottimeworld.moneyfamilyworldway time casestudyrightpeoplestudy.governmentnumbernumbermoney-yearchildcompanystudent-bookpointmoneyschoolcasewayroomcase.fact-rightlot";
        patternMatch = "government";
        patternNoMatch = "week";
        matches.add(87);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 189, was " + comparator.getComparisonCount(), 189 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 192, was " + comparator.getComparisonCount(), 192 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_33() {
        text = "roomlifeworldmoneyfamily night.family.night-area story thing factarea.waterbook thing studentfamilytimeroomsystem familyissue familypeoplestate-casenight home-studysystem-life.thingstatebookwater.group.familyworld pointmonth partlotprogramgroup childcountry";
        patternMatch = "student";
        patternNoMatch = "place";
        matches.add(86);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 268, was " + comparator.getComparisonCount(), 268 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 264, was " + comparator.getComparisonCount(), 264 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_34() {
        text = "right.pointmoneywaterwaterstudystate-worldfactlotsystemworldstate factissue.year story-year.programmoneyroompointroommoneynumberthing";
        patternMatch = "issue.";
        patternNoMatch = "people";
        matches.add(70);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 136, was " + comparator.getComparisonCount(), 136 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 137, was " + comparator.getComparisonCount(), 137 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_35() {
        text = "factmonthcase book-systemhomedaymoneyweek-factnight place-pointgovernmentstorymonthtime.month worldchildworldstory-pointmonth.familyweekgovernmentsystemstateyearnightstudent.childthing-nightpartstorycountry.areapeople-numbersystem wordissue family.nightfamilynumber.";
        patternMatch = "story";
        patternNoMatch = "school";
        matches.add(73);
        matches.add(109);
        matches.add(194);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 277, was " + comparator.getComparisonCount(), 277 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 280, was " + comparator.getComparisonCount(), 280 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_36() {
        text = "wordmonthprogramchildnumber group right case student grouppartweekmoneyweekgroup family way.factbook.studyissue.government statefact-factrightdaylifelife ";
        patternMatch = "child";
        patternNoMatch = "people";
        matches.add(16);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 158, was " + comparator.getComparisonCount(), 158 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 160, was " + comparator.getComparisonCount(), 160 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_37() {
        text = "study.statething issue-worldwaterhomesystemstudenttime childgovernmenthomecase.yearissuegroupissuearea-fact.issuebookwaterschoolthingtimestudycompanystudy-world";
        patternMatch = "study.";
        patternNoMatch = "family";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 176, was " + comparator.getComparisonCount(), 176 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 161, was " + comparator.getComparisonCount(), 161 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_38() {
        text = "worldnumber schoolmoneyyearcaseyear child.governmentbook pointmoneyschool";
        patternMatch = "world";
        patternNoMatch = "story";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 73, was " + comparator.getComparisonCount(), 73 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 76, was " + comparator.getComparisonCount(), 76 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_39() {
        text = "arealot.nighttimewordroom.factcompany";
        patternMatch = "fact";
        patternNoMatch = "year";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 37, was " + comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 37, was " + comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_40() {
        text = "issue-placestudentgovernmentworld countrygroup.roomschoolwatercountrypartschoolyear life-";
        patternMatch = "world ";
        patternNoMatch = "lot";
        matches.add(28);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 90, was " + comparator.getComparisonCount(), 90 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 94, was " + comparator.getComparisonCount(), 94 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_41() {
        text = "group-systemcompanynumberfamily dayareacompanypointstory-studentnumberprogram.water.childway-monthsystemstate.year-system.thingcasecountry word schoolway-pointnumber pointgroupfactissuelotmoney systempartmoney-";
        patternMatch = "system";
        patternNoMatch = "room";
        matches.add(6);
        matches.add(98);
        matches.add(115);
        matches.add(194);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 218, was " + comparator.getComparisonCount(), 218 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 224, was " + comparator.getComparisonCount(), 224 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_42() {
        text = "studyareahomeyearcountrynightcountry.fact.childsystemnumberstudy-governmentdaystory";
        patternMatch = "day";
        patternNoMatch = "place";
        matches.add(75);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 86, was " + comparator.getComparisonCount(), 86 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 83, was " + comparator.getComparisonCount(), 83 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_43() {
        text = "governmentpeoplelotrightbook.weekrightlifebook.worldroom";
        patternMatch = "lot";
        patternNoMatch = "fact";
        matches.add(16);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 59, was " + comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 57, was " + comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_44() {
        text = "studyweek.way.timenumber";
        patternMatch = "number";
        patternNoMatch = "month";
        matches.add(18);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_45() {
        text = "number systempeoplecase-areaworldplacestatelife peoplestudymoneywordnightstudentwordnightbookstudentthingstorycompanynight.child-numbersystem.money-roombookpartdayweekchildstateissueprogramlife-familylifelifewaterroomthingbookweekstatefamilytimepart";
        patternMatch = "day";
        patternNoMatch = "year";
        matches.add(160);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 257, was " + comparator.getComparisonCount(), 257 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 259, was " + comparator.getComparisonCount(), 259 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_46() {
        text = "issueroom-wateryear-number weeklotthingroom homeprogramfactmoneyfact-";
        patternMatch = "home";
        patternNoMatch = "family";
        matches.add(44);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 70, was " + comparator.getComparisonCount(), 70 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 70, was " + comparator.getComparisonCount(), 70 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_47() {
        text = "state schoolfact-factstatemoneyword year wordmoney-money-peoplestudenthomewordcountrygovernmentgroupplacethingstudentmoneyworldprogramsystemweek";
        patternMatch = "school";
        patternNoMatch = "room";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 150, was " + comparator.getComparisonCount(), 150 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 154, was " + comparator.getComparisonCount(), 154 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_48() {
        text = "companyfamily ";
        patternMatch = "family ";
        patternNoMatch = "people";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was " + comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_49() {
        text = "roomweekfamilyweek.book countrymoneywater-moneypoint-home-lot-partsystemwater.way";
        patternMatch = "point-";
        patternNoMatch = "right";
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 82, was " + comparator.getComparisonCount(), 82 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 86, was " + comparator.getComparisonCount(), 86 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_50() {
        text = "issuearea-wordyearprogramlife";
        patternMatch = "issue";
        patternNoMatch = "book";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_51() {
        text = "monthstory book.liferightarea worldplacefamilygrouppart water moneymoneystorylifeschoolgovernmentsystem-people.studentpointcaseroom-weeknightpoint.";
        patternMatch = "book.";
        patternNoMatch = "company";
        matches.add(11);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 147, was " + comparator.getComparisonCount(), 147 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 150, was " + comparator.getComparisonCount(), 150 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_52() {
        text = "people";
        patternMatch = "people";
        patternNoMatch = "family";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_53() {
        text = "thinggovernmentnight week governmentmoney.pointstory life.water monthmonthstory government school-bookstate.issuestudywayprogrampeople.world ";
        patternMatch = "water ";
        patternNoMatch = "student";
        matches.add(58);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 144, was " + comparator.getComparisonCount(), 144 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 148, was " + comparator.getComparisonCount(), 148 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_54() {
        text = "systemworldcase-nightroomroomstorystudy-people-waterwater issuemoney";
        patternMatch = "water";
        patternNoMatch = "school";
        matches.add(47);
        matches.add(52);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 69, was " + comparator.getComparisonCount(), 69 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 75, was " + comparator.getComparisonCount(), 75 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_55() {
        text = "word-thingissuefactlife.homepoint.wordroomrightcasepoint-worldweekstudyright.factgovernmentlotweekbook-";
        patternMatch = "room";
        patternNoMatch = "place";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 109, was " + comparator.getComparisonCount(), 109 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 105, was " + comparator.getComparisonCount(), 105 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_56() {
        text = "countryarea familyyear-family roomfamilynumber-peoplefamilyyearsystemschoolplacepart-point.pointissuegroupworldissuegovernment-wayplace-factsystem-life government yearlifeschoolyearhomestory number home.rightnumbertimenumberplace.money.area";
        patternMatch = "right";
        patternNoMatch = "word";
        matches.add(203);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 257, was " + comparator.getComparisonCount(), 257 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 242, was " + comparator.getComparisonCount(), 242 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_57() {
        text = "program-groupcompany.nightstorywaystudentworldstateareastoryweekhome";
        patternMatch = "state";
        patternNoMatch = "child";
        matches.add(46);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 71, was " + comparator.getComparisonCount(), 71 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 69, was " + comparator.getComparisonCount(), 69 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_58() {
        text = "studyhomewater";
        patternMatch = "home";
        patternNoMatch = "right";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_59() {
        text = "rightworldprogramhomestory-study lot waterrightnightwater.groupcompany.studentnumberlot way.wordcompanystudentplace.part-familyprogramcompanyfact righthome rightstudy.systemnightlotplacebooklife-arearightthing-world-lotcountryprogramwordweekworldcasefamily";
        patternMatch = "part-";
        patternNoMatch = "child";
        matches.add(116);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 265, was " + comparator.getComparisonCount(), 265 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 264, was " + comparator.getComparisonCount(), 264 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_60() {
        text = "factroomcompany study case-companything-worldplace lotwatermonthpeople-wordfactstaterightchild-daydayplacewater lifepeople-pointpartthingway-child.placegovernmenthomestudentwaterwater";
        patternMatch = "state";
        patternNoMatch = "money";
        matches.add(79);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 186, was " + comparator.getComparisonCount(), 186 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 189, was " + comparator.getComparisonCount(), 189 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_61() {
        text = "lifeissue-money.areapartissue thinghome stateright-issue-studentyear-roomplaceyeartime-studentday-grouppointprogram-homenumbersystemmonthschool.homestudyroomgovernmentstorypeoplewater.wordwaterstorywaterlot.government thing student thing area";
        patternMatch = "school.";
        patternNoMatch = "fact";
        matches.add(137);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 257, was " + comparator.getComparisonCount(), 257 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 243, was " + comparator.getComparisonCount(), 243 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_62() {
        text = "casepointtimewayhome-homestudent lot.programcompanyweekfactcountrywaygroupday-countrycompany.world peopleday.yearmoneychild.waterdayprogramtime ";
        patternMatch = "company.";
        patternNoMatch = "word";
        matches.add(85);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 150, was " + comparator.getComparisonCount(), 150 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 149, was " + comparator.getComparisonCount(), 149 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_63() {
        text = "issue-right country system number-programfamilypointstudyareastudent.story.wordprogramstudentroomthingsystemareamoney system-placepartyearstudent";
        patternMatch = "point";
        patternNoMatch = "water";
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 149, was " + comparator.getComparisonCount(), 149 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 146, was " + comparator.getComparisonCount(), 146 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_64() {
        text = "home-dayworldmonth-fact-casenumberwordstate monthfamilytimeword.statechildgovernment.countryissuegovernmentchildplacegovernmentbookpeoplerightnightgroup";
        patternMatch = "time";
        patternNoMatch = "money";
        matches.add(55);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 165, was " + comparator.getComparisonCount(), 165 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 161, was " + comparator.getComparisonCount(), 161 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_65() {
        text = "life-roomwayschoolprogramfamilynumber-world.studylot issuestudentstudent water.factroomway.roomlifenumber-system-student-night.room";
        patternMatch = "life";
        patternNoMatch = "part";
        matches.add(0);
        matches.add(95);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 135, was " + comparator.getComparisonCount(), 135 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 132, was " + comparator.getComparisonCount(), 132 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_66() {
        text = "point.worldprogram.grouplotschoolareanight world place.day-rightfactstudentsystem placestatesystemstateway.roomstudent case bookfact lifecasegroupstudystudentcasenumberprogram.programworldgovernmentbookcountrypeople case-systemwayhometime-";
        patternMatch = "program.";
        patternNoMatch = "company";
        matches.add(11);
        matches.add(168);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 247, was " + comparator.getComparisonCount(), 247 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 249, was " + comparator.getComparisonCount(), 249 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_67() {
        text = "childtimewayyearyear.roomgovernment.factsystemschoolyearstorymonthpeoplefamilymonth programplace moneycountrywayroompeople.worldfamilylife.program";
        patternMatch = "family";
        patternNoMatch = "case";
        matches.add(72);
        matches.add(128);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 148, was " + comparator.getComparisonCount(), 148 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 151, was " + comparator.getComparisonCount(), 151 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_68() {
        text = "groupschoolcompanypeoplehomelotschoolgovernmenttime.";
        patternMatch = "time.";
        patternNoMatch = "number";
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 58, was " + comparator.getComparisonCount(), 58 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 55, was " + comparator.getComparisonCount(), 55 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_69() {
        text = "numberprogram booktime-wordfact-area-place areanightwaywater.book.thingpointstudentfactworldweekwaypartnightissuecase.worldmoney studentnightmoney government";
        patternMatch = "thing";
        patternNoMatch = "month";
        matches.add(66);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 170, was " + comparator.getComparisonCount(), 170 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 162, was " + comparator.getComparisonCount(), 162 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_70() {
        text = "casegroupmoneynightpointwaybookyear-childstate-program-statepartissuepointbookgroupsystemplacegroupyear-way.state.";
        patternMatch = "book";
        patternNoMatch = "student";
        matches.add(27);
        matches.add(74);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 114, was " + comparator.getComparisonCount(), 114 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 121, was " + comparator.getComparisonCount(), 121 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_71() {
        text = "studentlotstoryroom.groupmoneystudent-night-homepeopledaymoney groupissuewayplacenumber.worldpeoplerightlife-area-company-program lifefamilystudypointnumbermonthchild";
        patternMatch = "family";
        patternNoMatch = "school";
        matches.add(134);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 168, was " + comparator.getComparisonCount(), 168 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 172, was " + comparator.getComparisonCount(), 172 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_72() {
        text = "studentprogramnumberworld";
        patternMatch = "student";
        patternNoMatch = "country";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_73() {
        text = "case water roomprogramlotareagovernmentcompanypeople.nighttimeprogram-student thingroomgovernmentwaystudentmoneythingcountrywaterstorygovernment";
        patternMatch = "program";
        patternNoMatch = "place";
        matches.add(15);
        matches.add(62);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 147, was " + comparator.getComparisonCount(), 147 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 149, was " + comparator.getComparisonCount(), 149 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_74() {
        text = "rightwayyear-areacasesystem-night";
        patternMatch = "system-";
        patternNoMatch = "company";
        matches.add(21);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 34, was " + comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_75() {
        text = "casetime story-yearpointbook-homestudy-";
        patternMatch = "year";
        patternNoMatch = "student";
        matches.add(15);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 40, was " + comparator.getComparisonCount(), 40 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 41, was " + comparator.getComparisonCount(), 41 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_76() {
        text = "schoolstudy.statestory";
        patternMatch = "story";
        patternNoMatch = "part";
        matches.add(17);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_77() {
        text = "lotthingthing-";
        patternMatch = "thing";
        patternNoMatch = "money";
        matches.add(3);
        matches.add(8);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_78() {
        text = "statestatehome system.studentschool-schoolhome-homelifewatergovernment.book studentstudy-numbernighttimefact roomlife-pointwaterareaweek.word.wayhomelifepartcompanyplace-weekstate.place-money.room areatime.studentstorypeopleissuemoney studypeople";
        patternMatch = "student";
        patternNoMatch = "country";
        matches.add(22);
        matches.add(76);
        matches.add(206);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 258, was " + comparator.getComparisonCount(), 258 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 252, was " + comparator.getComparisonCount(), 252 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_79() {
        text = "lifecompanydayworldtimewaycompany book areatime-";
        patternMatch = "time-";
        patternNoMatch = "word";
        matches.add(43);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 53, was " + comparator.getComparisonCount(), 53 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 50, was " + comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_80() {
        text = "studypartnumberpoint-statemoney.companymonthcompany country-roomarea studentright-yearfactstudent";
        patternMatch = "company ";
        patternNoMatch = "place";
        matches.add(44);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 100, was " + comparator.getComparisonCount(), 100 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 101, was " + comparator.getComparisonCount(), 101 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_81() {
        text = "stateprogram partissueyear-lot familyrightplace monthworldtime.childchildyearfamily-group.";
        patternMatch = "group.";
        patternNoMatch = "story";
        matches.add(84);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 97, was " + comparator.getComparisonCount(), 97 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 93, was " + comparator.getComparisonCount(), 93 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_82() {
        text = "thingfactwaydaystudentliferightroomwordplaceday moneywordcasefamilygovernment state-";
        patternMatch = "way";
        patternNoMatch = "water";
        matches.add(9);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 86, was " + comparator.getComparisonCount(), 86 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 87, was " + comparator.getComparisonCount(), 87 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_83() {
        text = "waterstudy";
        patternMatch = "water";
        patternNoMatch = "school";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_84() {
        text = "companyprogram.placeday water.wayrightlotstorymoney-monthstudentmoneyfactarearoomlife area-homegroupschoolstudymoneynumberwordlotday-daything";
        patternMatch = "day-";
        patternNoMatch = "government";
        matches.add(129);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 146, was " + comparator.getComparisonCount(), 146 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 144, was " + comparator.getComparisonCount(), 144 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_85() {
        text = "storynumber-";
        patternMatch = "story";
        patternNoMatch = "system";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_86() {
        text = "child";
        patternMatch = "child";
        patternNoMatch = "thing";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_87() {
        text = "factcompanymoneypartcompanypartstudyschoollifeway room-yearfactcompanynightwatercountry-case.yearareacountry money-childstudystudentright-storyareacountry childareaprogramsystemlot.childgroupright";
        patternMatch = "student";
        patternNoMatch = "state";
        matches.add(125);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 203, was " + comparator.getComparisonCount(), 203 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 204, was " + comparator.getComparisonCount(), 204 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_88() {
        text = "numbercase-nightlifepartmonth.schoolcompanysystem program.point.waterfactplaceissueparttime issuedayroom-childmoney.life.time childstudy state-grouproomlotbook rightschoolareachild issuesystemlot-government";
        patternMatch = "part";
        patternNoMatch = "thing";
        matches.add(20);
        matches.add(83);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 211, was " + comparator.getComparisonCount(), 211 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 223, was " + comparator.getComparisonCount(), 223 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_89() {
        text = "companysystemtimecasefamilyright programstudentplacegroup.week roomfamilymoney-timestorytimeprogrambookyearschoolhomepeople.governmentschoolchildtimeright.word";
        patternMatch = "school";
        patternNoMatch = "state";
        matches.add(107);
        matches.add(134);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 164, was " + comparator.getComparisonCount(), 164 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 166, was " + comparator.getComparisonCount(), 166 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_90() {
        text = "moneymoneypoint program-thingcountry-placestateweek";
        patternMatch = "thing";
        patternNoMatch = "fact";
        matches.add(24);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 55, was " + comparator.getComparisonCount(), 55 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 51, was " + comparator.getComparisonCount(), 51 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_91() {
        text = "storythingissue-childweekword.companyyearschoolhomewaynumber-number daystudent hometimegovernmentroomcompanystateprogramwayhomethingfact-moneyday-child";
        patternMatch = "thing";
        patternNoMatch = "place";
        matches.add(5);
        matches.add(127);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 159, was " + comparator.getComparisonCount(), 159 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 154, was " + comparator.getComparisonCount(), 154 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_92() {
        text = "wayday";
        patternMatch = "way";
        patternNoMatch = "state";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_93() {
        text = "family-dayfamilyrightmoney part.child-state-word.wordword";
        patternMatch = "child-";
        patternNoMatch = "home";
        matches.add(32);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 57, was " + comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 59, was " + comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_94() {
        text = "studypartmoney-programwordgroup-partroommonthwordstudent placecaseissuegroup family right.issue ";
        patternMatch = "family ";
        patternNoMatch = "government";
        matches.add(77);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 96, was " + comparator.getComparisonCount(), 96 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 100, was " + comparator.getComparisonCount(), 100 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_95() {
        text = "monthstudyarea companyfactrightprogram systemhomedaygovernmentplace-company-waymonth family-day placestateareatimeword yearissue monthpart-state-areapointlotcountrychildnumberplace-government system-night-nightweek";
        patternMatch = "right";
        patternNoMatch = "book";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 226, was " + comparator.getComparisonCount(), 226 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 215, was " + comparator.getComparisonCount(), 215 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_96() {
        text = "numbermoneybook day.governmentcase weekwordbookcompanyplace lotgovernmentrightdaycasewordfamily-number.word story.lotpart-month.groupgovernmentstudystudygroup wordbookfamilynight.room-areacountrypartnightworldway-roommoney-water";
        patternMatch = "room-";
        patternNoMatch = "point";
        matches.add(179);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 247, was " + comparator.getComparisonCount(), 247 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 234, was " + comparator.getComparisonCount(), 234 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_97() {
        text = "group.time-month.";
        patternMatch = "group.";
        patternNoMatch = "way";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 17, was " + comparator.getComparisonCount(), 17 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 17, was " + comparator.getComparisonCount(), 17 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_98() {
        text = "studypointhome-group thingright-lotnight.number.storyprogramlot-";
        patternMatch = "lot";
        patternNoMatch = "life";
        matches.add(32);
        matches.add(60);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 65, was " + comparator.getComparisonCount(), 65 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 66, was " + comparator.getComparisonCount(), 66 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_99() {
        text = "homebookweek.homeworld ";
        patternMatch = "book";
        patternNoMatch = "government";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testKMP_100() {
        text = "family.yearmonth-systemschoolnumber.homeworldstudygovernmentmonththingnumber.bookstory.word wordtime";
        patternMatch = "year";
        patternNoMatch = "way";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.kmp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 104, was " + comparator.getComparisonCount(), 104 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.kmp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 103, was " + comparator.getComparisonCount(), 103 >= comparator.getComparisonCount());
    }



    @Test(timeout = TIMEOUT)
    public void testRabinKarp_1() {
        text = "countrypart issuearea-system.monthprogramnumber-wordsystem money.areayear.country.case.study-roomfamilyworldstudenthomegroup area";
        patternMatch = "student";
        patternNoMatch = "book";
        matches.add(108);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_2() {
        text = "timefact.schoolfactday.countrywatercompany-nightyearlotrightbooksystemthinghome-nightpartrightstudypartpoint.state-rightmonth.studyright moneymoney-water state school-programcountryright-";
        patternMatch = "school";
        patternNoMatch = "room";
        matches.add(9);
        matches.add(160);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_3() {
        text = "moneymoneypart-dayyearmonthmoneyplace-nightsystem-numberpointfamilydayfamilyissue familyroomprogram studyyearbooklot placeworldchildstudentworldcountrynightsystemlife state family moneygroupschool.";
        patternMatch = "money";
        patternNoMatch = "fact";
        matches.add(0);
        matches.add(5);
        matches.add(27);
        matches.add(180);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was " + comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_4() {
        text = "numberwaterstudentpartcasewaycountry issueprogramdaymoneyweek rightgovernmentroomsystemrightstory.government-school number month fact-schoollotwordpeopleissuewaylifesystem-fact-peopleright.";
        patternMatch = "system-";
        patternNoMatch = "thing";
        matches.add(165);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_5() {
        text = "peoplepointsystem placegovernment waterpoint moneyhome daythingthingthingcase-homenumber.monthwaycaserightpartpart place wordwordarea.weeksystem-companywater book night rightareasystem pointlifeprogram.lifenightwayroomissueplaceright ";
        patternMatch = "water ";
        patternNoMatch = "study";
        matches.add(152);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_6() {
        text = "place-systemthingfact companystudycase-arealot-schoolyear yearweekbook-word";
        patternMatch = "year ";
        patternNoMatch = "part";
        matches.add(53);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_7() {
        text = "companyrightgovernmentchildmonthwayrightpart numbernumberstudystatepartschoolprogramstudy";
        patternMatch = "part";
        patternNoMatch = "story";
        matches.add(40);
        matches.add(67);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_8() {
        text = "moneytime-place-yearcasecountry.year lifearea state monthstudentcasemonth timenumber ";
        patternMatch = "money";
        patternNoMatch = "system";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_9() {
        text = "weekissuestudent-area.companywaywater-systemfactarea studentweekstudentnumberfamily worldchildfamilytimefamilycountrysystemgovernment.worldthing studentnumber statemonth-story";
        patternMatch = "fact";
        patternNoMatch = "group";
        matches.add(44);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_10() {
        text = "waywayarearightnumber-room-wordmoney case-program program.government.time";
        patternMatch = "number-";
        patternNoMatch = "point";
        matches.add(15);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_11() {
        text = "factlifewordthing homecase-companylifeplacemoneymoney.country-country place familycompanyprogramrightword nightwaynightareanight studyroom child right.people";
        patternMatch = "right.";
        patternNoMatch = "part";
        matches.add(145);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_12() {
        text = "programlotroom.schoolgovernment.nightweekwordway.studentwordhomegroupcompany-day.student bookcountry countrythingstudyarea";
        patternMatch = "room.";
        patternNoMatch = "right";
        matches.add(10);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_13() {
        text = "timepeoplethingstatechild pointfact-thingfacttimenightlot-companyroomstudyfactsystempart companycase.statecase-family-roomtime lotcompanypointschool.word-case group.thingtime.way.thing.placeday wordroomroomcountryword-case";
        patternMatch = "study";
        patternNoMatch = "right";
        matches.add(69);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_14() {
        text = "water-storysystem-yeargroupnightthing.systemweekright.homewayworldarea-studentfactyearpartworldrightbook wordgroup rightrightcountry-monthroombook-family.week thingmoneyschool-roompoint-point issuegroupstate.year monthbooktime";
        patternMatch = "word";
        patternNoMatch = "program";
        matches.add(105);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_15() {
        text = "bookcompanynumberhomestudentlifebookstoryroomnumber-systemlife.childwayweekroom-wordnightthing-factlot-thingmoneypoint familyhomemonthareatimecompany.schoolschoolfact.factnightyearlot company";
        patternMatch = "area";
        patternNoMatch = "program";
        matches.add(134);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_16() {
        text = "school.storychildgroupcasebook-book-time.case.time-moneygroupyearstudyareafamily";
        patternMatch = "money";
        patternNoMatch = "life";
        matches.add(51);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_17() {
        text = "childyeardaycountry.issuestudyprogramwaystudentchildfact.yearpartdayway.point.storycompany.part water.part-childthingthingfamilyhome-thingstatecountry";
        patternMatch = "thing";
        patternNoMatch = "word";
        matches.add(112);
        matches.add(117);
        matches.add(133);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_18() {
        text = "number bookworldstory-placetime-place home-waterwaterstudentroomright ";
        patternMatch = "water";
        patternNoMatch = "study";
        matches.add(43);
        matches.add(48);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_19() {
        text = "peoplemoneyhomeareastateschoolfamilyprogramfact-programyeargroupareastorystate issueschool book.thinggovernment number.factarearightareagroup homeparthome weekcompanybookstudy systemmoney.lot lot programstatewordhomefamilycasecountrywayissuecase-waterbook";
        patternMatch = "number.";
        patternNoMatch = "place";
        matches.add(112);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_20() {
        text = "roomsystemstoryareanight-placenight-place-countryschool-word studystudy year home-storything-countrymonth";
        patternMatch = "system";
        patternNoMatch = "way";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_21() {
        text = "pointissueschool moneywordpoint.groupstudypeople.nightchild factprogram-statecase-";
        patternMatch = "money";
        patternNoMatch = "student";
        matches.add(17);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_22() {
        text = "systemday lotbookmoney.storyfact rightchild storynight casestory.familylifeworldfact money numbergrouptimecompany peoplecompanylifepartlotcountrylot waybookmoney schoolwordissue.case-issuedaystudystatepartpeople.worldissueprogram";
        patternMatch = "right";
        patternNoMatch = "student";
        matches.add(33);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_23() {
        text = "schoolnumbernight.day.companyplaceright roomprogram moneystorymonth.point.studenttime-state school.fact-nightlifestate room-lotthing group.governmentprogrammonthstudentlifemoney.governmentyear world";
        patternMatch = "lot";
        patternNoMatch = "study";
        matches.add(124);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_24() {
        text = "programmonth wordcompany student.storyfamily";
        patternMatch = "month ";
        patternNoMatch = "study";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_25() {
        text = "placebookpartworld-schoolstatemoneyright-studentnumberparthome.wordyearstory-waterday room peoplestudy.storywaycountryplace-bookroom.nightrightroom.nightfact-thingwaygovernmentcountrypeoplelife.wordschoolthingmoneyworldlifepeopleroom";
        patternMatch = "money";
        patternNoMatch = "program";
        matches.add(30);
        matches.add(209);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_26() {
        text = "childnumberhomeprogram-study.program government";
        patternMatch = "program-";
        patternNoMatch = "area";
        matches.add(15);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_27() {
        text = "programplaceworddayareatimeissuecasewaterwater-homefact-familystatenight-peoplenumberstorynighthome.child.studentstudentstory-";
        patternMatch = "number";
        patternNoMatch = "school";
        matches.add(79);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_28() {
        text = "weekcase.daythinghomestate group point studypartweekissuecase.place.system-rightmonthfamilygroup-factlifeissue.student-casenumberworld time countrynight studentworldnumberthing-areaweekwaywatermoneynumbergovernmentrightschoolmoney.caseright waystudent timestudentcompany";
        patternMatch = "group-";
        patternNoMatch = "people";
        matches.add(91);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_29() {
        text = "homeroomgroupdaydaynightnumber-book student.childstudentcountry nightcompany";
        patternMatch = "home";
        patternNoMatch = "part";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_30() {
        text = "monthwater";
        patternMatch = "month";
        patternNoMatch = "program";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_31() {
        text = "group.wayschoolwaterpoint.year arealot schoolrightstorymonthareahomefact watergroup.lot.bookpointnightlotfamilyhomeprogramlifebook-waterpointgovernment";
        patternMatch = "point.";
        patternNoMatch = "room";
        matches.add(20);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_32() {
        text = "yearweek.program-lotnightweekfactlotcase lotwater";
        patternMatch = "year";
        patternNoMatch = "room";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_33() {
        text = "nightbookweekcase peoplelifetime studywatercasethingrightworldworldlotrightsystemthing-bookroom ";
        patternMatch = "water";
        patternNoMatch = "day";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_34() {
        text = "companytime-studentmonthfamilygovernmentgroupissuecountryroomissue study-waylot";
        patternMatch = "issue";
        patternNoMatch = "word";
        matches.add(45);
        matches.add(61);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_35() {
        text = "company-room.";
        patternMatch = "room.";
        patternNoMatch = "student";
        matches.add(8);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_36() {
        text = "timeschool groupworldnumber program-studyareapeopletimecompanyschool.story moneystudyright waystudentgroup point.lot";
        patternMatch = "company";
        patternNoMatch = "week";
        matches.add(55);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_37() {
        text = "issueroom night-state.childwaterstudentschoolpartbookbookword-familyrightroomcompanycaseworldcasefactpointcasecompanyschoolcasechildwordpointlifepeopleplace issue night number.part-timegovernment.rightweekmoney.yearnumbercase government.roomhomepointschool monthtime";
        patternMatch = "family";
        patternNoMatch = "country";
        matches.add(62);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_38() {
        text = "place-countrystory.casegroupgroup weekfamily lotareachild money.dayroom weekworldroomstoryworld.stateplaceyeardaygovernmentcountrywater.case-";
        patternMatch = "story.";
        patternNoMatch = "issue";
        matches.add(13);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_39() {
        text = "programpartwater groupnightworldcaseplace-rightplace-placepeopleschooltime-groupstudentgroupcountryareanumbercompany-family-casemonthpoint.night.statewayissuestateprogram";
        patternMatch = "world";
        patternNoMatch = "story";
        matches.add(27);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_40() {
        text = "countrycompanylifestudysystempoint-way";
        patternMatch = "way";
        patternNoMatch = "home";
        matches.add(35);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_41() {
        text = "storyhome.waystudyfamilynumberbooktimeway.place";
        patternMatch = "way.";
        patternNoMatch = "area";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_42() {
        text = "childissuenight.rightstorypartthingstorywayplacepeople monthsystem weekchild-lifelife.student-studentpartbookmonthplacelotweekworldgroupstudyareayeartime-company studystate.lot worldtime-issue.groupthingnumbernumber-peopleword.partlifepointarearight-country ";
        patternMatch = "part";
        patternNoMatch = "home";
        matches.add(26);
        matches.add(101);
        matches.add(227);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_43() {
        text = "way-booklotmonth moneycountry.roomyearnumberissueschool-lifelotstoryprogramfactlifepointschoolschool-programweek-money-programpart numbercountryday";
        patternMatch = "program";
        patternNoMatch = "family";
        matches.add(68);
        matches.add(101);
        matches.add(119);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_44() {
        text = "word-familywater-moneyday";
        patternMatch = "money";
        patternNoMatch = "area";
        matches.add(17);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_45() {
        text = "people-wordright thing.waterwaypart company-groupfactplacecompanynumber monthcompanystudy.numberissue.issue-worldissuehomegroupstudy.yearplace-room.roomyear-statenightwaythingway-school.wayweekwater.right.place-programpartcountry-worldstoryworld fact book lotway.";
        patternMatch = "room.";
        patternNoMatch = "family";
        matches.add(143);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_46() {
        text = "studyfamilycompanyissue.issue-lifegovernmentissue";
        patternMatch = "government";
        patternNoMatch = "state";
        matches.add(34);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_47() {
        text = "bookareanumber.weekarea-companyschool familybookhomecasetimechildroom wordlot.right programfamilylifestatestatemonththingmoneyhomecompany.room";
        patternMatch = "thing";
        patternNoMatch = "story";
        matches.add(116);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_48() {
        text = "issue.country right-homeareanumberfact-people world-studentchildschool-governmentcasegovernmentpointnumberchild.roomarea.bookcasestudy-companyhomepeoplefamily.systemcountrysystem-part moneypoint.governmentword.state-area placestudy.groupfamilymoneynumbergovernmentschoolschoolplace roomwater-day";
        patternMatch = "country ";
        patternNoMatch = "week";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_49() {
        text = "placestorygovernmentprogram state worldtimecase-moneycountryweek timeright-companylife.night-people world waterwordthingfamilyfamily.groupchildprogram.factprogram-system.pointstory-child";
        patternMatch = "world";
        patternNoMatch = "year";
        matches.add(34);
        matches.add(100);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_50() {
        text = "peoplepeoplenightareastudy yearcountrywordwaywaymoney school bookgovernmentmoneycountry roomgovernmenthomeprogram lotrightyearstudentgroup-case";
        patternMatch = "way";
        patternNoMatch = "issue";
        matches.add(42);
        matches.add(45);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_51() {
        text = "moneyfactstory nightday-caseplacerightmoneynumber.peoplefact studywayarea companywater-homesystem-home studenttimehomegovernment people people.";
        patternMatch = "people";
        patternNoMatch = "room";
        matches.add(50);
        matches.add(129);
        matches.add(136);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_52() {
        text = "week programnight programcompanycaseworldchildfacttime childsystem.factfamily-year.story-childstudy-week.casecountrywaycountryhomecase-pointgroupissue.money-state";
        patternMatch = "system.";
        patternNoMatch = "right";
        matches.add(60);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_53() {
        text = "nightlife government-nightcasegroupmoneystory lot.factyearmonth study wayrightprogram programbookpeoplepartworlddayprogramnumberlot-homeright.place-world.monthfamilycompanylifeyearyearwayarea-";
        patternMatch = "lot-";
        patternNoMatch = "week";
        matches.add(128);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_54() {
        text = "room-monthpartareastoryschoolmonthfamilyschool";
        patternMatch = "part";
        patternNoMatch = "day";
        matches.add(10);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_55() {
        text = "people.systemschoolcompanyschoolstudentrightstudy.liferoom family-month-waterword.case-familyfamilyway schoolnight";
        patternMatch = "month-";
        patternNoMatch = "story";
        matches.add(66);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_56() {
        text = "studystudent.yearbooktimeliferightissuemoney.studentfamilywordweek-governmentbookmoneyright schoolfamilyway-point daytime issuecasechildrightday.bookmoney world arealifebookpartlotplacecompanystatemoney.";
        patternMatch = "issue";
        patternNoMatch = "country";
        matches.add(34);
        matches.add(122);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_57() {
        text = "numbermoney.worldnumberwaywater-lifegroup.programsystem.world.month-peoplerightstoryweek peoplenightpointpeople-placestory.night-casesystemchildsystemstate world-wayworldstudent-thingpartstate.systemcaseroomlifeissuecompany-lotpointgovernment";
        patternMatch = "room";
        patternNoMatch = "word";
        matches.add(203);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_58() {
        text = "room.time group night week.child right.weekstory-program";
        patternMatch = "week";
        patternNoMatch = "government";
        matches.add(22);
        matches.add(39);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_59() {
        text = "wordwaypartthingmonth-day-time room-moneystateplaceright timegovernmentroommoneybook point-statepeople roomworld.country.life.country.countryright-governmentwordroombook.wordpointschool systempartwordwaterword monthway";
        patternMatch = "money";
        patternNoMatch = "week";
        matches.add(36);
        matches.add(75);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_60() {
        text = "day.room.peopleschool systemplaceyearcountryhomestateissuepartweek";
        patternMatch = "home";
        patternNoMatch = "money";
        matches.add(44);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_61() {
        text = "numbermonth-yearwordmoneycasepeoplemoney thingcompanything-part.familypointchild.groupstateissue.childright weekdayfact way-fact waywaterschoolissuestateareaweekwayfamilychildpoint country";
        patternMatch = "number";
        patternNoMatch = "study";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_62() {
        text = "wordyearyear thing-home ";
        patternMatch = "year";
        patternNoMatch = "part";
        matches.add(4);
        matches.add(8);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_63() {
        text = "time.system-pointprogrambook.worldschoolareamonthwater.countrypoint.case ";
        patternMatch = "program";
        patternNoMatch = "issue";
        matches.add(17);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_64() {
        text = "nightsystemnumbernumber-schoolpeopleissuestudentwaythingschoolplacecase issuenightthingstudy-childwater-program thing.night countryareamoneypeoplenight ";
        patternMatch = "people";
        patternNoMatch = "part";
        matches.add(30);
        matches.add(140);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_65() {
        text = "right programcasenightworldarea-statelife-schoolnightpeople.programyearwaterchildweekpointcasegrouplifelot.numbermoneypeopleprogrammoneyroom systemfactschoolwaterschoolcountrystoryhomestate-governmenthome governmentbookyear-number.";
        patternMatch = "point";
        patternNoMatch = "month";
        matches.add(85);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_66() {
        text = "weeknumberchildrightstatemoney money country homeprogramcompanystatechild-";
        patternMatch = "program";
        patternNoMatch = "world";
        matches.add(49);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_67() {
        text = "story-room-yeargroup-childplacecaseplacestudentschoolarea system.case room group family government.placesystemplace.part roomcaseyear-thing water-book";
        patternMatch = "room ";
        patternNoMatch = "point";
        matches.add(70);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_68() {
        text = "systemplace timelotfactschoolstory family time statemoneyplacemoneyway.government-water-week-wordstate.group-waycountry-time-studydaywordwaterroom.roompeople.thinggroup right";
        patternMatch = "time";
        patternNoMatch = "student";
        matches.add(12);
        matches.add(42);
        matches.add(120);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_69() {
        text = "casecompany.homeareawordroomwater-point program homethingmonth month monthfamily daychild placeyear-moneygrouppointtime-student";
        patternMatch = "water-";
        patternNoMatch = "lot";
        matches.add(28);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_70() {
        text = "worldstudynightpeople case thingfamilycountry lifeweekworld-monthmonthchildplace.storymonthfactcase wordhome-placemonthlife-groupnumbercompanydaychild.waystatestudenttime";
        patternMatch = "time";
        patternNoMatch = "lot";
        matches.add(166);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_71() {
        text = "moneyfamilystoryhomestudentfamilystudy.statemoney-company-worldstory.point companywaytimecountryareawaterpeoplemoneylotnight";
        patternMatch = "people";
        patternNoMatch = "system";
        matches.add(105);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_72() {
        text = "part-monthcountry-area right fact.country.homepeoplepartlot";
        patternMatch = "country.";
        patternNoMatch = "child";
        matches.add(34);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_73() {
        text = "systemlotdaystorygroupfamily.yearworldareamonthroomdayweekwaterprogram homeplacestudentsystemstatecasecase.part-storynumber schoolroom world-";
        patternMatch = "case.";
        patternNoMatch = "night";
        matches.add(102);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_74() {
        text = "factwaterwaystategovernment-caseschoolday waterfamilypeoplehomewater.worldworlddaypartyearpeople-weekhomeschool.";
        patternMatch = "day ";
        patternNoMatch = "book";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_75() {
        text = "country-yeargrouprightgroup dayfamilyworldstateparttimehomemoneyway";
        patternMatch = "country-";
        patternNoMatch = "word";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_76() {
        text = "area-factcountry-daypeoplefact";
        patternMatch = "fact";
        patternNoMatch = "night";
        matches.add(5);
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_77() {
        text = "peoplewatergroup-people.part-government-country-storymonth-groupchildprogramfactprogramgovernmenttimestory dayfamilystate-monthstudentword.student week-bookthingcasesystem student-rightnumber statedaydayplaceworldcompanyfamily monthplace";
        patternMatch = "company";
        patternNoMatch = "money";
        matches.add(213);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_78() {
        text = "area areaweekprogramwaymonth";
        patternMatch = "month";
        patternNoMatch = "thing";
        matches.add(23);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_79() {
        text = "placethinglotstorywaterwater stateissue governmentprogramcountrycountrygovernmentpeople systemwaymonthcompanyarearoomstatetimenight.peoplegovernmentstory companyissuestory yearstorygroupgroupschoolstudyissueyear-childmoneydaylifestudyschool";
        patternMatch = "issue ";
        patternNoMatch = "family";
        matches.add(34);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_80() {
        text = "placepeopleschoolyearbookgovernmentwaterareaday-wordmonthday childhomeweek-governmentwater.systemmoney areacompanynumber.wordplace-fact.placecountry-weekbookcompany-child-familycountrystate schoolsystem";
        patternMatch = "water.";
        patternNoMatch = "right";
        matches.add(85);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_81() {
        text = "studyschoolsystemlot.groupstudyday-month-issuefamily wayprogramareaprogramweek-roomgroup worldmoney-dayissueschoolmoney-homeschoolsystem.word.casethingword.time-life ";
        patternMatch = "world";
        patternNoMatch = "fact";
        matches.add(89);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_82() {
        text = "family-yearchildpoint";
        patternMatch = "year";
        patternNoMatch = "lot";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_83() {
        text = "student-yearstudy timetime-moneyfacttimegovernmentway.week studentlot daythingschoolyear.statething-";
        patternMatch = "time";
        patternNoMatch = "world";
        matches.add(18);
        matches.add(22);
        matches.add(36);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_84() {
        text = "programsystemgovernmentcompanypartpeople-people.programroomcompanyworldthingrightissueliferightareastory nightcaselotday.moneyhomenightfamilyworldissuestudychildprogrampeoplepoint right-wayhome lifewaterway statething-weekworldplaceroomworld ";
        patternMatch = "money";
        patternNoMatch = "fact";
        matches.add(121);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_85() {
        text = "day.thinghome student wordnumberlot systemrightpeople-life.worldwater systemissuelifewaymonthplacecountrybooklife-programmonthwordprogramnight.day lifeissuepoint-month.lifenight money.countrymoneyworld lotway";
        patternMatch = "country";
        patternNoMatch = "week";
        matches.add(98);
        matches.add(184);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_86() {
        text = "monthfactstudent programpart-yearchildpartarea placelotschooltimelife-caselot.pointwatergrouphome.governmentpointstudythingday-monthstudent.wayyear-casefamilylotcountry-caselot lot.rightfamilyplace.caseworldcountrypeoplewayroom.companylotmoney";
        patternMatch = "fact";
        patternNoMatch = "book";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_87() {
        text = "waterlotcaseissue student-governmentprogramgovernmentarea-weekright.family government.way.areamonthfactwayfactbook.companythingstudent.nightplace-lotstory waypointpeople dayissue system wordgovernmentyear-lifeprogram-issuestory";
        patternMatch = "day";
        patternNoMatch = "state";
        matches.add(170);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_88() {
        text = "case groupweekarealifestate pointword governmentstorynumber";
        patternMatch = "point";
        patternNoMatch = "school";
        matches.add(28);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_89() {
        text = "wayschoolstory.place.water night home story.companywater studyway.time.issueschoolstorynightgroup nightstudentstorychildissuesystemlife.word-world";
        patternMatch = "story.";
        patternNoMatch = "point";
        matches.add(9);
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_90() {
        text = "school countryyearfamilytime world people caseroomcasestory-studypoint areawater worldyearmoneynumber.placecase.roomwater-government-";
        patternMatch = "water ";
        patternNoMatch = "week";
        matches.add(75);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_91() {
        text = "peopleday child-thingtimecasecountrybookparthomemonthpeoplenightplace.weektime.place-area";
        patternMatch = "month";
        patternNoMatch = "water";
        matches.add(48);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_92() {
        text = "casepointfamilyfact case.issuewaymonthrightschool.way factstudent-storysystem.issue-week.group.moneyprogram storylotprogrampointyear-lot story-numberroom thingstudy.";
        patternMatch = "program";
        patternNoMatch = "government";
        matches.add(100);
        matches.add(116);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_93() {
        text = "studyweekcompany monthpeoplestate worldschool-storylifemoney moneystorytimebookprogram-wayday.people-statestatemonthcaseplace-bookpointcompanywayareastateweekareaplacenight.place";
        patternMatch = "school-";
        patternNoMatch = "word";
        matches.add(39);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_94() {
        text = "right-thingfamilyarearight.people-arearoomschoolgroupcasehomewater-liferoomcompanymonthissueissue student stateweek.caseworldstatesystem bookpeople childcasenightfamily.groupareaword-state";
        patternMatch = "area";
        patternNoMatch = "study";
        matches.add(17);
        matches.add(34);
        matches.add(174);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_95() {
        text = "waterwaterroomstatecase-numberworldstory-casemonth systemareastategovernment room ";
        patternMatch = "water";
        patternNoMatch = "point";
        matches.add(0);
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_96() {
        text = "issue-homeissuegovernment-bookgroup-wordnightstorynumberfamily-year month-lifeareahometimecompanyroomissuegroupstoryfamilyrightnightstudent-point-governmentpartstudentthing schoolhomechild.studywaystoryplace governmentrightyearbookwaywordissuewater ";
        patternMatch = "life";
        patternNoMatch = "week";
        matches.add(74);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_97() {
        text = "nightlotright.school-wordtime-homechild homerightroomplace peopleprogrammonthplace timewater yearchildchild yearschool homestudent pointsystem.peopletimeplaceweek";
        patternMatch = "system.";
        patternNoMatch = "fact";
        matches.add(136);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_98() {
        text = "lotpeople.world-casefamilyissue grouppointwaterday-moneymonthfamily thing-pointwordnumberbookpointtimefact-homechildgovernmentpeoplecase.schoolrightareaarearightbookfamily.monthcountryworldnumberroom room-monthchild.thing";
        patternMatch = "point";
        patternNoMatch = "program";
        matches.add(37);
        matches.add(74);
        matches.add(93);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_99() {
        text = "monthlotstory group.world-";
        patternMatch = "lot";
        patternNoMatch = "student";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testRabinKarp_100() {
        text = "homefamilycaselotprogramlotarea night.family study placenightroommonthwater timecase-lot.rightstoryarea familylifebookrightcompany-people year-peoplegovernment-issue pointlot.roomnumber-companyway-country";
        patternMatch = "way-";
        patternNoMatch = "school";
        matches.add(193);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.rabinKarp(patternMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.rabinKarp(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 0, was " + comparator.getComparisonCount(), 0 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_1() {
        text = "factstudentrightstorylottime.nightstudystudywaterprogramwater-school placewaymonthpart-moneystudentright-";
        patternMatch = "place";
        patternNoMatch = "family";
        matches.add(69);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was " + comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_2() {
        text = "nightroomgovernmentprogramwaycompany thingsystem.yearthingfamily weekmonthcompanydaymonththing case wordpart-place governmentprogram life systempoint number-water-issueright.yearschoolgrouplifefamilyhomeright.wordlot-lotplace ";
        patternMatch = "group";
        patternNoMatch = "area";
        matches.add(184);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 55, was " + comparator.getComparisonCount(), 55 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 68, was " + comparator.getComparisonCount(), 68 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_3() {
        text = "way wayparttimeplacerightweekroom";
        patternMatch = "way";
        patternNoMatch = "study";
        matches.add(0);
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_4() {
        text = "lotfamilycountry governmenttimeyear-studentprogram programyearyearstorygroup-casewordstate group";
        patternMatch = "word";
        patternNoMatch = "area";
        matches.add(81);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_5() {
        text = "way system";
        patternMatch = "system";
        patternNoMatch = "point";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_6() {
        text = "right.programchildstudent area timerightroomgovernmentwordnumberweekchild.moneyhomedayfamilyroom-bookstoryfactstudy-yearstorygroup case";
        patternMatch = "area ";
        patternNoMatch = "point";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_7() {
        text = "book.fact area state.companything-childyeartimeworld story.book.place.partgroup part waterrightcompanymonthbookfact partchild.companyweekissuestudyissue.areanumberissuestudyschool governmentlot governmentlifecasestudyrightmonthcasenight";
        patternMatch = "fact ";
        patternNoMatch = "family";
        matches.add(5);
        matches.add(111);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 65, was " + comparator.getComparisonCount(), 65 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_8() {
        text = "companymoneypeoplefact-countryplacepart-life student numberpeople-day";
        patternMatch = "company";
        patternNoMatch = "government";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_9() {
        text = "studentroomworld-systemstudyareaprogramstatelife-companyissueprogram-studypartplace roomday systemcompany wordgrouppointstudentchildlifewordcase roomhome room.familypartstate.childareanumbermoneyday wayissuelot";
        patternMatch = "company ";
        patternNoMatch = "book";
        matches.add(98);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was " + comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 57, was " + comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_10() {
        text = "monthchildmonthcasechildright-homecompany-homeplaceprogram familyword.day issue moneylottimeplacegroupcase-word.place day-group.numbercountryyear booksystemrightissueplacepeoplestoryhomepeopleschoolchild companytime.government-";
        patternMatch = "place";
        patternNoMatch = "night";
        matches.add(46);
        matches.add(92);
        matches.add(112);
        matches.add(166);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 75, was " + comparator.getComparisonCount(), 75 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 57, was " + comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_11() {
        text = "child-word.waymoney.casefamilysystemcompany-homeroompart.watergovernmentweek program.moneywaywordwater story companyareacountryschool thing ";
        patternMatch = "week ";
        patternNoMatch = "month";
        matches.add(72);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_12() {
        text = "roomschool right.righthomecasesystem-place.bookstudy-familyday.systemwordcompanymoneyhomecountrynumberbookmoney.programworldworld fact storychildstateworld.";
        patternMatch = "case";
        patternNoMatch = "week";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 47, was " + comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_13() {
        text = "waydaypointgrouparealifemonthissuegovernment worldbook";
        patternMatch = "way";
        patternNoMatch = "company";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_14() {
        text = "studentyear.room thingfactsystem.studenthome-childareacountrystaterightstudent-studystudyarea governmentmonthright.timething-time homelotworld";
        patternMatch = "time";
        patternNoMatch = "part";
        matches.add(115);
        matches.add(125);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 51, was " + comparator.getComparisonCount(), 51 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_15() {
        text = "wordpeoplecasepartschoolareaschoolstudy program.partsystem.nightway.storylotmoney-storycountrygovernment-statepoint worldarea.point";
        patternMatch = "way.";
        patternNoMatch = "student";
        matches.add(64);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 26, was " + comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_16() {
        text = "week.placeweekfactweekcountryfact-booktimelifeissueweekdaynumbernight.stateissue.governmentstudent part.life factmonth-weeksystem-moneypartplace government.watergovernment timethingwaterprogramstudentplacecountrypeoplemoneycountry monthnightlotworldrighttimechild factpart ";
        patternMatch = "fact";
        patternNoMatch = "room";
        matches.add(14);
        matches.add(29);
        matches.add(109);
        matches.add(264);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 98, was " + comparator.getComparisonCount(), 98 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 78, was " + comparator.getComparisonCount(), 78 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_17() {
        text = "way-student-homeday.companywaterpartgroupwatertimetimestudy-";
        patternMatch = "time";
        patternNoMatch = "system";
        matches.add(46);
        matches.add(50);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_18() {
        text = "rightmonthsystemnight companyfact thingfact facttime.wateryear.numberfamilyhomecompanynumber company roomlifeday fact.thingsystem.peopleyear partwatertimeissue.life.childbookweekfamilymonthcompanysystemway";
        patternMatch = "number ";
        patternNoMatch = "story";
        matches.add(86);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 49, was " + comparator.getComparisonCount(), 49 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_19() {
        text = "grouppeopleday.moneyprogrampointfamily year.system-country studyfamilypoint.";
        patternMatch = "point";
        patternNoMatch = "area";
        matches.add(27);
        matches.add(70);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was " + comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_20() {
        text = "programchildway-systemstatelifecompanystatemonth thing thingwaterfamily nightgroup moneygovernmentlife groupfactweekthingnightwayprogramprogramstorypeopleworld.word-countrynightcase day";
        patternMatch = "month ";
        patternNoMatch = "time";
        matches.add(43);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 44, was " + comparator.getComparisonCount(), 44 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 55, was " + comparator.getComparisonCount(), 55 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_21() {
        text = "storything.night-issuefactpoint familylotstorygovernmenthome familynightmoneyyear.governmentnight";
        patternMatch = "point ";
        patternNoMatch = "time";
        matches.add(26);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 38, was " + comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_22() {
        text = "familyfact-thing system-book case country rightpointstate book-placeroom";
        patternMatch = "right";
        patternNoMatch = "home";
        matches.add(42);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was " + comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was " + comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_23() {
        text = "time month family-bookworldgovernmentbookhomelife worldwatercompanygrouplotstudentstudent year.day";
        patternMatch = "life ";
        patternNoMatch = "week";
        matches.add(45);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_24() {
        text = "familybookgrouptimeplaceday story.area familyfact-wordlotmoneyfamily homenightfact.companyday-issue.life-student-world";
        patternMatch = "life-";
        patternNoMatch = "system";
        matches.add(100);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 24, was " + comparator.getComparisonCount(), 24 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_25() {
        text = "watercase-areabookpart number-partschoolsystemweek-word.year-group-numberpartlotyear.factgroupschoolworld parthomepeopleworldpointcountryyearnumberway.thing room storynightword-part case thing.number governmentwaterword.waterrightstatecasedayfamily water.system-";
        patternMatch = "part";
        patternNoMatch = "time";
        matches.add(18);
        matches.add(30);
        matches.add(73);
        matches.add(106);
        matches.add(177);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 99, was " + comparator.getComparisonCount(), 99 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 76, was " + comparator.getComparisonCount(), 76 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_26() {
        text = "system wordright childissue";
        patternMatch = "word";
        patternNoMatch = "month";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_27() {
        text = "partwordmonthdaystoryworldweek company-thingthingprogramwayschoolhome";
        patternMatch = "part";
        patternNoMatch = "group";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_28() {
        text = "studentroom yearmonthroom-thingprogramstorythingfamilycompanyfamily factnightissuelot nightschool.";
        patternMatch = "story";
        patternNoMatch = "money";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 28, was " + comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_29() {
        text = "thingfactnumbermonth schoollifeworldnight-groupcountry.bookweeknightfactissuehome-countrywater.point.way";
        patternMatch = "point.";
        patternNoMatch = "system";
        matches.add(95);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 26, was " + comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_30() {
        text = "story ";
        patternMatch = "story ";
        patternNoMatch = "case";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 1, was " + comparator.getComparisonCount(), 1 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_31() {
        text = "numberthing.governmentmoney pointstudyhome pointfact.night case-worldcountry-night.thingcasemonththingwordfactnumber.monthstudent.casewater.lot.weekmoneystudy.lotstudybookstudywordhomesystemhomehomeworldpoint placemonthgrouprightweek.issuechild";
        patternMatch = "book";
        patternNoMatch = "part";
        matches.add(167);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 70, was " + comparator.getComparisonCount(), 70 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 67, was " + comparator.getComparisonCount(), 67 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_32() {
        text = "part.child-areafact-companygroupthing-group-lotchild-programwaychildthing-issue companyroom-wordword-number rightprogram-government";
        patternMatch = "lot";
        patternNoMatch = "life";
        matches.add(44);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 53, was " + comparator.getComparisonCount(), 53 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 34, was " + comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_33() {
        text = "numbercompanystory.room-grouprightroom.night-peoplefamily.pointthinglifegroupnumberweek-people-dayfamily point ";
        patternMatch = "family.";
        patternNoMatch = "system";
        matches.add(51);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 25, was " + comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 21, was " + comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_34() {
        text = "familymonthcasegovernment-money countrywordcompany.governmentpartrightrightgroupprogram-child-child book lifechildlife book-part.group-companynightgrouplife factfamily room.yearpartdaycompanystudysystemyearcasenumber";
        patternMatch = "case";
        patternNoMatch = "place";
        matches.add(11);
        matches.add(206);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 67, was " + comparator.getComparisonCount(), 67 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 59, was " + comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_35() {
        text = "school.student-factstatestudy student.month programnumberway.night";
        patternMatch = "study ";
        patternNoMatch = "people";
        matches.add(24);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 17, was " + comparator.getComparisonCount(), 17 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_36() {
        text = "school-government";
        patternMatch = "government";
        patternNoMatch = "study";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_37() {
        text = "governmentbook roomtimestorynumberweekwayfamilyplacestudy-school-roomyearweek school word thing-money-familyfamily familyday.word.casenightweek.money";
        patternMatch = "case";
        patternNoMatch = "group";
        matches.add(130);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 46, was " + comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_38() {
        text = "lifestory.water.numberprogramareahome companypeopleplace.waterfactnumber-waycase areatimegovernment-";
        patternMatch = "case ";
        patternNoMatch = "lot";
        matches.add(76);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_39() {
        text = "program storystudent thingschoolmonthareawater.familycompany groupissue issueweekpointpartroom companyareagovernmenthome ";
        patternMatch = "company";
        patternNoMatch = "case";
        matches.add(53);
        matches.add(95);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 38, was " + comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 39, was " + comparator.getComparisonCount(), 39 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_40() {
        text = "systemwayplacegovernment.study ";
        patternMatch = "place";
        patternNoMatch = "lot";
        matches.add(9);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_41() {
        text = "nightthing-issuestudychildwordnumber ";
        patternMatch = "number ";
        patternNoMatch = "state";
        matches.add(30);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_42() {
        text = "timeroom.homeschool.lotchild";
        patternMatch = "lot";
        patternNoMatch = "day";
        matches.add(20);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_43() {
        text = "week.factwayfamilymoneythingwater.state.wayschool schoolbook worldway life way familyschool-money issuemonthnightdayworld-waterbook.worldlot-word-familytimecompany facttime.state.way.";
        patternMatch = "way";
        patternNoMatch = "case";
        matches.add(9);
        matches.add(40);
        matches.add(66);
        matches.add(75);
        matches.add(179);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 81, was " + comparator.getComparisonCount(), 81 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 53, was " + comparator.getComparisonCount(), 53 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_44() {
        text = "week.moneygovernmentfamilydayweek partmonthright.lot.groupschool-homeissuefamilywater case government.roomstudyareastudy-roomprogram peoplelifechild-daylotpoint";
        patternMatch = "lot.";
        patternNoMatch = "system";
        matches.add(49);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was " + comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_45() {
        text = "pointissuechildpeopleprogram lifechildcompanymoney.areafamily.program-system.week";
        patternMatch = "system.";
        patternNoMatch = "world";
        matches.add(70);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was " + comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_46() {
        text = "storystudyroom.statemoney casetimegovernment.governmentschool.nightyearday programsystem.companyday peoplenumberpart.rightcompany partyeartime water number.number-wayarea-thinghome-rightbook-government.point school childcountrystudentstoryworldpeople areaweekmonthyearway ";
        patternMatch = "time";
        patternNoMatch = "lot";
        matches.add(30);
        matches.add(138);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 93, was " + comparator.getComparisonCount(), 93 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 114, was " + comparator.getComparisonCount(), 114 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_47() {
        text = "numberstudentprogram.childnighttimecase statenumber.nightgovernment.homecasetimeweekprogramcompanystorysystem caseroom.thing.countrybookroomstudent countryyearsystempart wayworld.study part bookroomgovernmentareagrouplot.rightfact.student";
        patternMatch = "group";
        patternNoMatch = "life";
        matches.add(212);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 61, was " + comparator.getComparisonCount(), 61 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 68, was " + comparator.getComparisonCount(), 68 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_48() {
        text = "pointpartfactissue-week.lotcase-year school.bookmoney.factpeople-waycase placeprogram-government.wayrightfactschoolstudent childroomfamilygrouppart casestudentday";
        patternMatch = "day";
        patternNoMatch = "company";
        matches.add(159);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 64, was " + comparator.getComparisonCount(), 64 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_49() {
        text = "monthpartroomfamily-areathing familystudent-roomfamilycompanyyear-fact point weektimelifebookprogram-study-study place";
        patternMatch = "study-";
        patternNoMatch = "people";
        matches.add(101);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_50() {
        text = "studynightstudentarea-rightstorygovernmentworldarea.storydayworldstudentarea.number companymonth-year";
        patternMatch = "world";
        patternNoMatch = "word";
        matches.add(42);
        matches.add(60);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_51() {
        text = "month partroomstatestudent yearwordcountrychildnightcompanyfamilymoneything right-issueword studentcase casecountrylifeissue way-yearwordnight-moneywordstudent governmentchild-state dayschoolmoneywordpartrightworldstudy.statepeople";
        patternMatch = "child-";
        patternNoMatch = "number";
        matches.add(170);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 50, was " + comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 46, was " + comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_52() {
        text = "wordgroup.waycasegovernmentlotsystemgovernmentworldpartweek-peopleyearroom-waygroupplace-placepartstudy.roomwordwater night-state.country";
        patternMatch = "government";
        patternNoMatch = "time";
        matches.add(17);
        matches.add(36);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 37, was " + comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_53() {
        text = "caseissueplacewater-rightright.night student-government-moneystudent-company peoplegroupsystemworldbookwater-year-money-number";
        patternMatch = "government-";
        patternNoMatch = "school";
        matches.add(45);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 24, was " + comparator.getComparisonCount(), 24 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_54() {
        text = "companywordprogrammonth-government schoolcaserightroomstudychildfact.way timefact issuestudystudyroomstorylotgovernment.way";
        patternMatch = "room";
        patternNoMatch = "water";
        matches.add(50);
        matches.add(97);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 44, was " + comparator.getComparisonCount(), 44 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_55() {
        text = "thingschoolmonthwayareabookyear.schoolpointpointnumberlifecase-people.bookdaystudentcountry.lotmonth-yearday-number-storynumbermoneyweek money";
        patternMatch = "school";
        patternNoMatch = "system";
        matches.add(5);
        matches.add(32);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 39, was " + comparator.getComparisonCount(), 39 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_56() {
        text = "life-fact-groupschoolstoryhomepart stateyearsystemnightstate bookcompanygovernmentissue-programpartthing lotfamilywordstory-room.programbookweekhomeareabook.country way weekgovernment-study";
        patternMatch = "fact-";
        patternNoMatch = "people";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was " + comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_57() {
        text = "wayhome book fact water-roompointday-weekfamily";
        patternMatch = "fact ";
        patternNoMatch = "right";
        matches.add(13);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was " + comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was " + comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_58() {
        text = "word areaweekweek familyroom-moneystudentlotchildlifepeoplelotlife-year-countrycaseright thing.wayfact program worldschoolsystemfamilytime-schoolchildwaterstory.weekdaypartstorybookarea ";
        patternMatch = "time-";
        patternNoMatch = "month";
        matches.add(134);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 52, was " + comparator.getComparisonCount(), 52 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_59() {
        text = "number year rightyearcompany issueplace-areastate governmentpeople lotissuestudy ";
        patternMatch = "place-";
        patternNoMatch = "fact";
        matches.add(34);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_60() {
        text = "numberway.word.right world area waystory-companyissuewaystatewaywaything lifeareastory";
        patternMatch = "life";
        patternNoMatch = "group";
        matches.add(73);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 26, was " + comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 18, was " + comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_61() {
        text = "week numberworldwatergroup factbookwateryeargroupstudent-worldweek.year-statenightbookfamilypeople.roomschoolpeople.areastudent thingprogramissue-water";
        patternMatch = "world";
        patternNoMatch = "part";
        matches.add(11);
        matches.add(57);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 45, was " + comparator.getComparisonCount(), 45 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 46, was " + comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_62() {
        text = "area-statenightrightsystemwaterpointfactfactgroupchild area rightstatemonth";
        patternMatch = "fact";
        patternNoMatch = "book";
        matches.add(36);
        matches.add(40);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was " + comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_63() {
        text = "casecompanydaystudyplace-system monthgroup.month.yearrightstory government countryrightissuecountry.storystorynightcompany";
        patternMatch = "night";
        patternNoMatch = "money";
        matches.add(110);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_64() {
        text = "yearpart-issuefactyearcompanypeople systempoint familycompany governmentprogramissuecompanypointword.nightroomissuefactwater.school issuestatecasepeople.school-book lot.governmentcountrychild-storyissue.world.water government.book";
        patternMatch = "issue";
        patternNoMatch = "month";
        matches.add(9);
        matches.add(79);
        matches.add(110);
        matches.add(132);
        matches.add(197);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 78, was " + comparator.getComparisonCount(), 78 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 54, was " + comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_65() {
        text = "fact.wordbook-way familyworld.childfactmonth-thingbookissueweek programcountrybooknumberstorymonth.issue.peoplestorygroupcountrycompanylotprogramrightweek timefactmonth monthpointwordmoneymonthchildcountry.lifeissuelifeyearcountry way-programworld peoplefact";
        patternMatch = "year";
        patternNoMatch = "home";
        matches.add(219);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 81, was " + comparator.getComparisonCount(), 81 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 82, was " + comparator.getComparisonCount(), 82 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_66() {
        text = "point.money-case.countryprogramyear.monthhomesystemnightpeople.groupgovernment.yearchild-monthday word.world companynight.systemnight right-companygrouprightthingweekplacebookprogramstudy issuechildthingfamily group school companyplacelife waychild";
        patternMatch = "government.";
        patternNoMatch = "lot";
        matches.add(68);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 38, was " + comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 97, was " + comparator.getComparisonCount(), 97 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_67() {
        text = "roomcountryplacecompanyfamily-companynumberissue-bookdaystudypointissueworldnumber government";
        patternMatch = "number";
        patternNoMatch = "money";
        matches.add(37);
        matches.add(76);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 30, was " + comparator.getComparisonCount(), 30 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was " + comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_68() {
        text = "homebookweekplace-roompart.numberpointstudentmoneything.system-number placeplace.caseroomstatewayhomebook student timestudentstatehomecasestudy-programpeople.people-monthstateword-homefactfactstudy-childpart";
        patternMatch = "people.";
        patternNoMatch = "year";
        matches.add(151);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 61, was " + comparator.getComparisonCount(), 61 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_69() {
        text = "groupthing roomissuechildlot";
        patternMatch = "child";
        patternNoMatch = "government";
        matches.add(20);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_70() {
        text = "waterhomeissue-worldfactmonthfactpartfactpointstate.programstudy wordwater-countryyearstudentmonthstudent.watergovernmentwordstudentstorystorybooknightworld time-thingpeoplefamilymoneybook";
        patternMatch = "fact";
        patternNoMatch = "life";
        matches.add(20);
        matches.add(29);
        matches.add(37);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 63, was " + comparator.getComparisonCount(), 63 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 54, was " + comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_71() {
        text = "money-story issueweek-issueprogram-roomsystemgroupprogramgovernmentgovernment.groupissuegroupareachildrightthing child-waymonthpartpointthingsystem";
        patternMatch = "child-";
        patternNoMatch = "number";
        matches.add(113);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 32, was " + comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 33, was " + comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_72() {
        text = "family ";
        patternMatch = "family ";
        patternNoMatch = "home";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 1, was " + comparator.getComparisonCount(), 1 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_73() {
        text = "placeschool timehome lotfactyearlifenumber-studenttime-child-studentworldchild.storystudentbooksystemissuestudy-water-countrygrouptimeprogramtimemoneypointright";
        patternMatch = "point";
        patternNoMatch = "room";
        matches.add(150);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 46, was " + comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_74() {
        text = "weekroom";
        patternMatch = "week";
        patternNoMatch = "group";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_75() {
        text = "numberprogrampart-area-pointgroup worldchildwordnight.familyroomplace.partrighttime thingschool weekpartlifeworldthingschool right group.waterschool-school.state-point company statenight.daylot";
        patternMatch = "night.";
        patternNoMatch = "people";
        matches.add(48);
        matches.add(181);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 52, was " + comparator.getComparisonCount(), 52 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_76() {
        text = "timedayissue.groupschool storyarealife.";
        patternMatch = "day";
        patternNoMatch = "week";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 16, was " + comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_77() {
        text = "schoolwordareapeople.place waterday.statesystemwater numberarea-study.night-";
        patternMatch = "system";
        patternNoMatch = "case";
        matches.add(41);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 22, was " + comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 26, was " + comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_78() {
        text = "homenumberfactday studygovernment case governmentsystem.placewater student.case-rightrightstudylotword.partwordnumberlifeissue.year worldpointwordcountry.people-";
        patternMatch = "year ";
        patternNoMatch = "state";
        matches.add(127);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 45, was " + comparator.getComparisonCount(), 45 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_79() {
        text = "place worldfactpartwater-story factschoolcompanyrightpoint.worldfamily issuearea countryissuewaynumberstorycase moneyrightlot-lotpeopletime.studywayplacebookfamilyhomecountrypartplacepartmoney";
        patternMatch = "story";
        patternNoMatch = "program";
        matches.add(25);
        matches.add(102);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 59, was " + comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_80() {
        text = "companymonthyearstorystudywordday peoplemoney nightpeople-child partstateprogram.pointcountry weekfamilyprogrampeopleroom-groupstorylot";
        patternMatch = "part";
        patternNoMatch = "school";
        matches.add(64);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 27, was " + comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_81() {
        text = "placeplace moneymoneypartissueschoolmoneypointcompanywaterprogram group issuefamilystudyplaceyearfamilynumberpoint lifewordmonthword-study areadaystorymonth governmentlifemonth.";
        patternMatch = "point ";
        patternNoMatch = "home";
        matches.add(109);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 38, was " + comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 60, was " + comparator.getComparisonCount(), 60 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_82() {
        text = "familyfact.";
        patternMatch = "family";
        patternNoMatch = "group";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_83() {
        text = "place rightlifecase-dayweek-areacountry part-lifewordmoney-daystudent roomnumbergovernmentmonth childlotfactroomgovernmentweekbookmoney studylot point studentschool-factpointthingarea-";
        patternMatch = "right";
        patternNoMatch = "family";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 50, was " + comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_84() {
        text = "companymonthnumber-book groupcompanystudent-";
        patternMatch = "number-";
        patternNoMatch = "case";
        matches.add(12);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was " + comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_85() {
        text = "governmentweekgovernmentchild-countrynightdaycaseprogramcompany group-right people-point.yearareawayyearstatelotyearwayweek-programword-countryworldchildgovernment timemonth.storyroomfamilycompanybookwaterbookplace.issueroom systemworldfactweekhome weekmonth";
        patternMatch = "government ";
        patternNoMatch = "life";
        matches.add(153);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 75, was " + comparator.getComparisonCount(), 75 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_86() {
        text = "factpartstudy issuenumberplacethingroomwater part homelot waterissue lifestategroupstudyyearwordworld-monthwater-roomdaypoint-moneygrouproomworldwatersystem.month-";
        patternMatch = "water";
        patternNoMatch = "night";
        matches.add(39);
        matches.add(58);
        matches.add(107);
        matches.add(145);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 58, was " + comparator.getComparisonCount(), 58 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was " + comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_87() {
        text = "school-companyweekgroup issue-lifemonthstudent.monthcompany-world-world-thingchildweekmonth program month.programnight wordtimegroupnightyear";
        patternMatch = "school-";
        patternNoMatch = "country";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 29, was " + comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 26, was " + comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_88() {
        text = "night-daywordcaserightpart.point ";
        patternMatch = "case";
        patternNoMatch = "issue";
        matches.add(13);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was " + comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_89() {
        text = "part issue-governmentsystem";
        patternMatch = "issue-";
        patternNoMatch = "thing";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_90() {
        text = "booklot.issuepeople";
        patternMatch = "lot.";
        patternNoMatch = "room";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 8, was " + comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 5, was " + comparator.getComparisonCount(), 5 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_91() {
        text = "child study";
        patternMatch = "study";
        patternNoMatch = "group";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was " + comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 2, was " + comparator.getComparisonCount(), 2 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_92() {
        text = "water issuewatermoneyprogramplaceworld.groupweekprogramstate-storyday.life partgroup.nightway areachildwaypointwaystudent.familydaygovernmentsystemlotgovernment";
        patternMatch = "money";
        patternNoMatch = "study";
        matches.add(16);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was " + comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_93() {
        text = "program.day-case.life systemtimeschoolbook part-storyareachild ";
        patternMatch = "time";
        patternNoMatch = "government";
        matches.add(28);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 23, was " + comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was " + comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_94() {
        text = "month systemstateword week childfact number-waysystemmonthpartgovernmenttimechilddayprogramworldmonthweekstory yearfacttimeroom programweekstoryissueplacetimegrouppeoplewordnumberhomedaystudylife worldtimefact";
        patternMatch = "program";
        patternNoMatch = "right";
        matches.add(84);
        matches.add(128);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 54, was " + comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 48, was " + comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_95() {
        text = "areastate.numberyear thingchildstory.storyday.people water.timenumberstate-pointcompanystudytimeprogram-timegovernment lotwater-familyyearcasenightareapeople week";
        patternMatch = "people ";
        patternNoMatch = "word";
        matches.add(46);
        matches.add(151);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was " + comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 43, was " + comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_96() {
        text = "study-family year-";
        patternMatch = "family ";
        patternNoMatch = "world";
        matches.add(6);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was " + comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 3, was " + comparator.getComparisonCount(), 3 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_97() {
        text = "rightpeopleweek place storypartwaterissue year-yearstudy pointnightprogramschool-systemstorygovernmentlot-wordstoryissuewayfacthomeschoolwordyearhomebooklifemoneyareathingdaygrouphomeworldstudent life nightpartstorynight.people-fact-placeroommonthweek";
        patternMatch = "water";
        patternNoMatch = "number";
        matches.add(31);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 73, was " + comparator.getComparisonCount(), 73 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 54, was " + comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_98() {
        text = "worldareaissuepartweek.roomlifecompanyroomdaystudy-nightgovernmentstudent-place night.weekroomrightstudynumberthingmoneycompanywaywaystory.program.caseworldpart-";
        patternMatch = "room";
        patternNoMatch = "lot";
        matches.add(23);
        matches.add(38);
        matches.add(90);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 59, was " + comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 64, was " + comparator.getComparisonCount(), 64 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_99() {
        text = "year.weekwaylifestudy";
        patternMatch = "life";
        patternNoMatch = "thing";
        matches.add(12);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was " + comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 4, was " + comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testBoyerMoore_100() {
        text = "system.people-year time moneyprogramchildbook-governmentrightworldpart.homelotcasegroupstory-group.factdayfamilycasesystemweek issuecompanyroomcase student";
        patternMatch = "case";
        patternNoMatch = "place";
        matches.add(78);
        matches.add(112);
        matches.add(143);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMoore(patternMatch, text, comparator));
        assertTrue("Comparison count should be 56, was " + comparator.getComparisonCount(), 56 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMoore(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 44, was " + comparator.getComparisonCount(), 44 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_1() {
        if (skipGalil) return;
        text = "familywordyearstudentstudygroup";
        patternMatch = "family";
        patternNoMatch = "story";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was "+comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 11, was "+comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_2() {
        if (skipGalil) return;
        text = "day-placestudent lifegovernmentpeopleprogramhomehome home-fact-";
        patternMatch = "day-";
        patternNoMatch = "month";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 22, was "+comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was "+comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_3() {
        if (skipGalil) return;
        text = "issuepeople-governmentthing stateschool-factlife.student-study.rightpartstate pointfamilygroup-familyright night-thing room.timecompany money.student.childcountrysystem ";
        patternMatch = "country";
        patternNoMatch = "program";
        matches.add(155);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was "+comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was "+comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_4() {
        if (skipGalil) return;
        text = "time.nightcase.studysystemwayright issueyearroom.pointword.";
        patternMatch = "time.";
        patternNoMatch = "day";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 23, was "+comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 22, was "+comparator.getComparisonCount(), 22 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_5() {
        if (skipGalil) return;
        text = "word.point issuepeoplewaything-areadayplace.nightroomword yearweekyearnightstoryroom-thinggroup issueprogram";
        patternMatch = "year";
        patternNoMatch = "time";
        matches.add(58);
        matches.add(66);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 44, was "+comparator.getComparisonCount(), 44 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 33, was "+comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_6() {
        if (skipGalil) return;
        text = "countrysystemwordwaterpartplacecompanyright";
        patternMatch = "company";
        patternNoMatch = "home";
        matches.add(31);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was "+comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 16, was "+comparator.getComparisonCount(), 16 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_7() {
        if (skipGalil) return;
        text = "schoolpeoplebook-countrynight-study school area-countryword-weekfactnightnightmoneystudent childpartplacemonth-homecompanyschool-familypart.waterareanumberright story storypartcase.wordpoint-way.numberpointwaystatestory ";
        patternMatch = "right ";
        patternNoMatch = "room";
        matches.add(155);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 56, was "+comparator.getComparisonCount(), 56 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 73, was "+comparator.getComparisonCount(), 73 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_8() {
        if (skipGalil) return;
        text = "thingwater waternumber people-areacountry-studylotsystemareabookpoint groupbook.timeroomfactcompanyissuelifeschoolareastudentpeople bookway factnumbergroup";
        patternMatch = "people-";
        patternNoMatch = "case";
        matches.add(23);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 40, was "+comparator.getComparisonCount(), 40 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 45, was "+comparator.getComparisonCount(), 45 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_9() {
        if (skipGalil) return;
        text = "company";
        patternMatch = "company";
        patternNoMatch = "home";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 13, was "+comparator.getComparisonCount(), 13 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 4, was "+comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_10() {
        if (skipGalil) return;
        text = "wordhome monthsystemrightpeople.casefamily-word.nightstudentwaterbookworldstorytimenightpeopleschoolgroup-familyrightgovernmentright arearight-waterprogramprogram.water week.numberstatepart monthwaymonthplaceprogram.wordfamilyplacemoneyroom";
        patternMatch = "week.";
        patternNoMatch = "lot";
        matches.add(169);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 60, was "+comparator.getComparisonCount(), 60 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 101, was "+comparator.getComparisonCount(), 101 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_11() {
        if (skipGalil) return;
        text = "place-moneygroup moneymoneypeopleway-company.casepeoplerightweek.companysystemstudentpartworldyearhomeissuegovernmentnight.partnight lifestudentwordmonthpoint roomtimewaterpeople.month.people.placecasegroup.study yearwayhomecasecaseroomprogramstudystateweek";
        patternMatch = "money";
        patternNoMatch = "day";
        matches.add(6);
        matches.add(17);
        matches.add(22);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 80, was "+comparator.getComparisonCount(), 80 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 103, was "+comparator.getComparisonCount(), 103 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_12() {
        if (skipGalil) return;
        text = "rightgovernment";
        patternMatch = "right";
        patternNoMatch = "thing";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was "+comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 8, was "+comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_13() {
        if (skipGalil) return;
        text = "place program moneystudenthomelotworldwordcountry-studentstudy room-moneywater grouprightpeoplenightfamilyprogramlot.money.booknightnightroom-studentwordyearwaterwordgovernment-state";
        patternMatch = "night";
        patternNoMatch = "month";
        matches.add(95);
        matches.add(127);
        matches.add(132);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 63, was "+comparator.getComparisonCount(), 63 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 50, was "+comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_14() {
        if (skipGalil) return;
        text = "monthcountrything.story-waystudy weekpeople.water.group.point state lotroomworldfactstudent-booklifeword.way homewordfamilysystem-";
        patternMatch = "state ";
        patternNoMatch = "right";
        matches.add(62);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 38, was "+comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was "+comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_15() {
        if (skipGalil) return;
        text = "arearightright word programnumberwaywater.systemplace-weekrightword study.lifebook.school.monthgroup-book.partpart.company.child-wordwaychildhomeschoolroomcompany.home childschoolfactsystemplace";
        patternMatch = "company.";
        patternNoMatch = "people";
        matches.add(115);
        matches.add(155);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 41, was "+comparator.getComparisonCount(), 41 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_16() {
        if (skipGalil) return;
        text = "moneyrightplacewordplacecasechildtimeplacewaycase.issue-systemwayprogramnight-state statelotnumberstatetimepoint timehomecompanylifepeopleareabookstudentcountryprogramroomstudentareapeoplechildtimeareaworldnumber-time-countrychildfactfactpeopleway.place";
        patternMatch = "right";
        patternNoMatch = "government";
        matches.add(5);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 67, was "+comparator.getComparisonCount(), 67 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 50, was "+comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_17() {
        if (skipGalil) return;
        text = "water.schoolcompanyprogram money-familywaterstudent-money-peopledaychild yearweekgroup-nightmonthstudylotcountryyearpointfactschoolpart.caselifeyear-world-program countrypoint thing-statethingwordwaypeople-factsystemplacething ";
        patternMatch = "people";
        patternNoMatch = "number";
        matches.add(58);
        matches.add(199);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 64, was "+comparator.getComparisonCount(), 64 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_18() {
        if (skipGalil) return;
        text = "studycasegroup companyroomroomchildthingroom-right";
        patternMatch = "room";
        patternNoMatch = "family";
        matches.add(22);
        matches.add(26);
        matches.add(40);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 31, was "+comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 15, was "+comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_19() {
        if (skipGalil) return;
        text = "schoolgrouppartareapart-familylifething schoolgovernment caseroom waycase.area";
        patternMatch = "thing ";
        patternNoMatch = "time";
        matches.add(34);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 25, was "+comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 24, was "+comparator.getComparisonCount(), 24 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_20() {
        if (skipGalil) return;
        text = "casenightplacefact-thingmoneygovernmentchild.month-story-point.thingstudenttime state case.weekpart factbookschool-issueway.home student-study-worldstudytimeareacompanyplace.right";
        patternMatch = "time ";
        patternNoMatch = "water";
        matches.add(75);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 58, was "+comparator.getComparisonCount(), 58 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_21() {
        if (skipGalil) return;
        text = "numberstorywaycompany.thingcompanystudentchildword book company.peopleworld-money.way";
        patternMatch = "child";
        patternNoMatch = "government";
        matches.add(41);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 25, was "+comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 18, was "+comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_22() {
        if (skipGalil) return;
        text = "waterchild-areaweekmonth year book.point-right areaplacewaybook";
        patternMatch = "area";
        patternNoMatch = "life";
        matches.add(11);
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 33, was "+comparator.getComparisonCount(), 33 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 23, was "+comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_23() {
        if (skipGalil) return;
        text = "grouprightweek-lotpointpeopleareagroupnumbertime-fact-booklifehomefamily-life room booktimeprogramareapart timestorywaterpartschoolplace ";
        patternMatch = "time";
        patternNoMatch = "night";
        matches.add(44);
        matches.add(87);
        matches.add(107);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 53, was "+comparator.getComparisonCount(), 53 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 38, was "+comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_24() {
        if (skipGalil) return;
        text = "wayright.family-issue.water wordmonth.placepeople-issue thing companypartbooklot familyroom-program-moneynumberpeoplewordhome.rightway";
        patternMatch = "place";
        patternNoMatch = "world";
        matches.add(38);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 39, was "+comparator.getComparisonCount(), 39 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was "+comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_25() {
        if (skipGalil) return;
        text = "family partday schoolrightpoint-homepart part-roomplace thing-casepartdayschool.wayworldday.pointnumberweekstorymoney-issue.issuedayworld-";
        patternMatch = "school";
        patternNoMatch = "student";
        matches.add(15);
        matches.add(73);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 42, was "+comparator.getComparisonCount(), 42 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was "+comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_26() {
        if (skipGalil) return;
        text = "family companyyear-storymonth.study-timesystemchildmonthcountrycompany-issue.parthome-governmentschool-place.monthpart.";
        patternMatch = "month.";
        patternNoMatch = "program";
        matches.add(24);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 32, was "+comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 28, was "+comparator.getComparisonCount(), 28 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_27() {
        if (skipGalil) return;
        text = "worldroomworldgrouphomestorynumber-student group-case lifeyearsystemwaycompany way water.dayworldlifefactlotweekhome.state-";
        patternMatch = "home";
        patternNoMatch = "place";
        matches.add(19);
        matches.add(112);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 46, was "+comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 31, was "+comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_28() {
        if (skipGalil) return;
        text = "year.weekyearcase-placeyear rightbook-";
        patternMatch = "year";
        patternNoMatch = "month";
        matches.add(0);
        matches.add(9);
        matches.add(23);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 23, was "+comparator.getComparisonCount(), 23 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 12, was "+comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_29() {
        if (skipGalil) return;
        text = "companymoneyplaceworldnight.issuegovernmentnightstudyprogrampart nighthomegovernmentnumberyearcompanylifepeoplenumberwaybookweekplace-right-thingstudentstudent-placerightgroupcompanyfactschool.countrybookfactstatewayschool.lot.areamoney-programmonththingcaselifecountry country.";
        patternMatch = "place-";
        patternNoMatch = "system";
        matches.add(128);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 66, was "+comparator.getComparisonCount(), 66 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 61, was "+comparator.getComparisonCount(), 61 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_30() {
        if (skipGalil) return;
        text = "fact people-waterhomeweektime-day-companytimewaterworldnumberpointthingstudy.study.area.placeplace areastudy-factfamily.companybookstudentmoney-";
        patternMatch = "study.";
        patternNoMatch = "system";
        matches.add(71);
        matches.add(77);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 43, was "+comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 37, was "+comparator.getComparisonCount(), 37 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_31() {
        if (skipGalil) return;
        text = "company-roompointtime peoplemoneyarea roomyear";
        patternMatch = "area ";
        patternNoMatch = "right";
        matches.add(33);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 20, was "+comparator.getComparisonCount(), 20 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 15, was "+comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_32() {
        if (skipGalil) return;
        text = "systemdaywaterdaymonth issue-lot.daycountrypartlifeyearstorywater.rightpointstategrouprightway";
        patternMatch = "state";
        patternNoMatch = "thing";
        matches.add(76);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 36, was "+comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 26, was "+comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_33() {
        if (skipGalil) return;
        text = "point-waythingstorywordnightpointworldhomebook systembookhome place familyrighttimewordnight stateyear placerightdayfamily studyroom ";
        patternMatch = "word";
        patternNoMatch = "school";
        matches.add(19);
        matches.add(83);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 46, was "+comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 31, was "+comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_34() {
        if (skipGalil) return;
        text = "pointyearpoint-water.peopleroom.bookgroup-stateright.year-factweekfact.partissue.daymonthwayyear-numberissuefact";
        patternMatch = "day";
        patternNoMatch = "country";
        matches.add(81);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 27, was "+comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_35() {
        if (skipGalil) return;
        text = "monthtimehomestudyyearyear.yearstate.monthmoney-word-number peoplechildmonth";
        patternMatch = "year.";
        patternNoMatch = "issue";
        matches.add(22);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 26, was "+comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 25, was "+comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_36() {
        if (skipGalil) return;
        text = "studentmonthnumberworldnumberpointpeoplewaynight";
        patternMatch = "month";
        patternNoMatch = "home";
        matches.add(7);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 18, was "+comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 19, was "+comparator.getComparisonCount(), 19 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_37() {
        if (skipGalil) return;
        text = "thing ";
        patternMatch = "thing ";
        patternNoMatch = "country";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 11, was "+comparator.getComparisonCount(), 11 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was "+comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_38() {
        if (skipGalil) return;
        text = "school nightroomright-nightmoney.lotstory.country-monthword.weekyearnumberwatercountryworldarea water system-placestate-peoplenumber wordhomemonthplacecasepointworldwaterplacehomeyear.program group-waycase-number-week";
        patternMatch = "school ";
        patternNoMatch = "issue";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 49, was "+comparator.getComparisonCount(), 49 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 58, was "+comparator.getComparisonCount(), 58 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_39() {
        if (skipGalil) return;
        text = "bookchild rightyearcase";
        patternMatch = "case";
        patternNoMatch = "study";
        matches.add(19);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 12, was "+comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 9, was "+comparator.getComparisonCount(), 9 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_40() {
        if (skipGalil) return;
        text = "student lifearea-storyissuecase.bookhome lotgovernmentchildthingfact programnightyeardayworldgroupword word";
        patternMatch = "student ";
        patternNoMatch = "room";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 31, was "+comparator.getComparisonCount(), 31 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 32, was "+comparator.getComparisonCount(), 32 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_41() {
        if (skipGalil) return;
        text = "grouparea-waygovernmentlife-studycountry-weekfamily.money-studentpartfamilynighttime-arearoompeoplepeople-lotsystemwordareaworldyearwordnumberyearstudystudy statestudygrouptimeworldthing";
        patternMatch = "study";
        patternNoMatch = "child";
        matches.add(28);
        matches.add(146);
        matches.add(151);
        matches.add(162);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 62, was "+comparator.getComparisonCount(), 62 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 49, was "+comparator.getComparisonCount(), 49 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_42() {
        if (skipGalil) return;
        text = "placenight-partpointstudywaterpointpointpart";
        patternMatch = "point";
        patternNoMatch = "student";
        matches.add(15);
        matches.add(30);
        matches.add(35);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 26, was "+comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 17, was "+comparator.getComparisonCount(), 17 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_43() {
        if (skipGalil) return;
        text = "issue.moneybookprogramwater monthbooksystemnight.thingcase-daygovernmentroomday-family-company.rightstudent-right-governmentfact-daylottime";
        patternMatch = "book";
        patternNoMatch = "word";
        matches.add(11);
        matches.add(33);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 49, was "+comparator.getComparisonCount(), 49 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 43, was "+comparator.getComparisonCount(), 43 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_44() {
        if (skipGalil) return;
        text = "timegroupbook-yearchild-book lotpeopleweekpeople school wayroom.lotareaweekthingstudypartchildcountryfact study number system word.numberplacenumber nighttimeplace family-bookroom";
        patternMatch = "group";
        patternNoMatch = "home";
        matches.add(4);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 46, was "+comparator.getComparisonCount(), 46 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 57, was "+comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_45() {
        if (skipGalil) return;
        text = "familywatergroupwaygovernmentgovernmentworld-partbookpartcasechildfact.watercountry-companyhome place-studyhome story nightfamily.waycase week childfamily-month-countrynight-time waterschool.area.student programlife ";
        patternMatch = "water";
        patternNoMatch = "people";
        matches.add(6);
        matches.add(71);
        matches.add(179);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 67, was "+comparator.getComparisonCount(), 67 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_46() {
        if (skipGalil) return;
        text = "areastate room-daysystem-partmoneyright case.roomgovernmentissue studywaterprogrambook right lifepointword-part countrytimerightpartfactstudent";
        patternMatch = "room";
        patternNoMatch = "number";
        matches.add(10);
        matches.add(45);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 50, was "+comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was "+comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_47() {
        if (skipGalil) return;
        text = "systemfamily home-wordarealifestorycompanypoint-child-wordareanumber-government.world-lotpeople-bookday-story statelotnumberplace.week.number groupgovernmentrightcountrymonthcountry-governmentrightprogram story storyyearchild homegovernment";
        patternMatch = "day-";
        patternNoMatch = "issue";
        matches.add(100);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 71, was "+comparator.getComparisonCount(), 71 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 59, was "+comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_48() {
        if (skipGalil) return;
        text = "programpeopleschool family childschoolgroup peopleschoolsystembook way.weekthing-systemcase-company-governmentlotchilddaylotbookdaycasegroupcountrystudent.country schoolnight.lifeschoolnumberschoolthingstateweek wayyearpart companypeoplecountrynightworldtimepeoplehome";
        patternMatch = "lot";
        patternNoMatch = "place";
        matches.add(110);
        matches.add(121);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 117, was "+comparator.getComparisonCount(), 117 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 74, was "+comparator.getComparisonCount(), 74 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_49() {
        if (skipGalil) return;
        text = "studentway groupstudy.studentyearareamoneypointrightcountry-place.life child-country";
        patternMatch = "right";
        patternNoMatch = "family";
        matches.add(47);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 26, was "+comparator.getComparisonCount(), 26 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 21, was "+comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_50() {
        if (skipGalil) return;
        text = "countrystateissuenumberrightlifemoneytimebook waystudytimelothomewaycase-systemwordstudy waterpointyearstatenumberroomwaychild lot-timechildtime schoolnightwaterlife countrything month.company year studentstate.lot.caseday";
        patternMatch = "year";
        patternNoMatch = "part";
        matches.add(99);
        matches.add(193);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 76, was "+comparator.getComparisonCount(), 76 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 72, was "+comparator.getComparisonCount(), 72 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_51() {
        if (skipGalil) return;
        text = "country.pointfamilygroup.pointmoneysystemstudentcountrywaterfact nightplace.month-worldstate-storyyearword.issuebooklotstate.timeprogram.areamoneysystemnumber homecase-group";
        patternMatch = "night";
        patternNoMatch = "right";
        matches.add(65);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 50, was "+comparator.getComparisonCount(), 50 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 51, was "+comparator.getComparisonCount(), 51 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_52() {
        if (skipGalil) return;
        text = "bookprogramyear";
        patternMatch = "year";
        patternNoMatch = "room";
        matches.add(11);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 10, was "+comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 6, was "+comparator.getComparisonCount(), 6 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_53() {
        if (skipGalil) return;
        text = "lotplace-timetimepartdaygovernmentfamilypartworldnightcase-companysystem-word-thingcountryhomeareatimebook.groupstate";
        patternMatch = "system-";
        patternNoMatch = "program";
        matches.add(66);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 35, was "+comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was "+comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_54() {
        if (skipGalil) return;
        text = "companyway.world.righttimeplacetimeprogramright.peoplenumberchildareafamilywaterliferighthomecountryprogramroomfactfactpeopleyear bookstoryareacountry.part month.";
        patternMatch = "life";
        patternNoMatch = "government";
        matches.add(80);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 58, was "+comparator.getComparisonCount(), 58 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 27, was "+comparator.getComparisonCount(), 27 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_55() {
        if (skipGalil) return;
        text = "lotschoolstate issuemonthlifepointwaterwordschoolcase.peoplefamilychildcaseyearwatergovernmentwaterchildfact numberright.point-right night placerightworldstudent.company-";
        patternMatch = "case";
        patternNoMatch = "system";
        matches.add(49);
        matches.add(71);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 59, was "+comparator.getComparisonCount(), 59 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 41, was "+comparator.getComparisonCount(), 41 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_56() {
        if (skipGalil) return;
        text = "stateprogramstory-yearmonthpointlife.wordwordchildchildtimelifeweekgroupmoneyweeklifethingbookwater programwayissue numbercasefactfamily-nightgovernmentlife wordstudentthingsystem-wayareabook part.month-schoolstudentissuemoney.nightfact studentprogramroom";
        patternMatch = "government";
        patternNoMatch = "country";
        matches.add(142);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 57, was "+comparator.getComparisonCount(), 57 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 56, was "+comparator.getComparisonCount(), 56 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_57() {
        if (skipGalil) return;
        text = "homelotroomstate study word-nightdaynumberstory-rightstorystudent.pointpart-moneypointgroupcase.world issuemoney-storyyearnumberpart.statetimeworldpeoplefact.daygroupdayworldlifechild thing numberschoolworldgovernmentpeople-lot.pointpart story";
        patternMatch = "money-";
        patternNoMatch = "area";
        matches.add(107);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 55, was "+comparator.getComparisonCount(), 55 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 77, was "+comparator.getComparisonCount(), 77 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_58() {
        if (skipGalil) return;
        text = "book.schoolnightpeoplegroup study.companyyearcountry-government life.familystorywayprogram systemstate storywaystory.systemchildissue.peoplefactfactworldchild-worldcase-week-point thing-wordroomstudyprogramstudy.life-wayfact ";
        patternMatch = "case-";
        patternNoMatch = "right";
        matches.add(164);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 63, was "+comparator.getComparisonCount(), 63 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 60, was "+comparator.getComparisonCount(), 60 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_59() {
        if (skipGalil) return;
        text = "dayweekyear room.studentworld";
        patternMatch = "world";
        patternNoMatch = "group";
        matches.add(24);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 15, was "+comparator.getComparisonCount(), 15 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 10, was "+comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_60() {
        if (skipGalil) return;
        text = "bookgroupweek-caselotwordmonth-roomway-liferight-thinglotschool.programprogramyearschoolwater-moneystudentwordnightchildhomeworldcase.number";
        patternMatch = "thing";
        patternNoMatch = "issue";
        matches.add(49);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 39, was "+comparator.getComparisonCount(), 39 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was "+comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_61() {
        if (skipGalil) return;
        text = "studybook-waystudenttime numbercompany-worldstorymoney-pointweek-systempointstudent.governmentcompany.worldrightstudent.familycountryhomepeoplecase.pointstatewaternighttimeissuewaterplace day-moneyyearwaterstatenight right worldchildmonth homeweekpoint.water-issuepoint-issue-";
        patternMatch = "day-";
        patternNoMatch = "fact";
        matches.add(188);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 87, was "+comparator.getComparisonCount(), 87 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 85, was "+comparator.getComparisonCount(), 85 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_62() {
        if (skipGalil) return;
        text = "moneythingareathingpartwatermonth.factcountrymoneydaycountry-night.placegovernmenthomenumberrightyearwater-time";
        patternMatch = "money";
        patternNoMatch = "life";
        matches.add(0);
        matches.add(45);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 38, was "+comparator.getComparisonCount(), 38 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was "+comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_63() {
        if (skipGalil) return;
        text = "timenumberpeople.stateyear life-countrytimestorymoneycompanytimeplace studentissuemonthmonth-countrystudent-moneywater roomwordcase-partissuegovernmentbookwaterlifesystem-nightstorypartpeople thingcasepart-governmentstory government issuestorystudent";
        patternMatch = "system-";
        patternNoMatch = "week";
        matches.add(164);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 64, was "+comparator.getComparisonCount(), 64 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 74, was "+comparator.getComparisonCount(), 74 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_64() {
        if (skipGalil) return;
        text = "bookrightpeopleway.systemweekword people.lotissue-rightroom.year-timeprogrampeople.night.bookplaceroomstorylifeplace-study state.word.familystudy";
        patternMatch = "right";
        patternNoMatch = "home";
        matches.add(4);
        matches.add(50);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_65() {
        if (skipGalil) return;
        text = "day-rightprogramfactwatercompany.program night.groupyearmoneystorypeople.study";
        patternMatch = "company.";
        patternNoMatch = "place";
        matches.add(25);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 25, was "+comparator.getComparisonCount(), 25 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 21, was "+comparator.getComparisonCount(), 21 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_66() {
        if (skipGalil) return;
        text = "life.numberareaschoolschoolstatestudystory program.month.programpeople-countrycountry-bookhomewaycountryday-issueday weekgroupwaterpartsystemgroup student roomissuebookcountrylifecaseway-state worddayyeartime issueplace point-month";
        patternMatch = "year";
        patternNoMatch = "government";
        matches.add(200);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 83, was "+comparator.getComparisonCount(), 83 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_67() {
        if (skipGalil) return;
        text = "weekareanumber moneysystem.peoplearea";
        patternMatch = "number ";
        patternNoMatch = "word";
        matches.add(8);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 18, was "+comparator.getComparisonCount(), 18 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 12, was "+comparator.getComparisonCount(), 12 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_68() {
        if (skipGalil) return;
        text = "area.way-systemnight";
        patternMatch = "system";
        patternNoMatch = "child";
        matches.add(9);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was "+comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 8, was "+comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_69() {
        if (skipGalil) return;
        text = "storystorystudy.student state-statechildweek.issue-storyissuechild-homestateword.child.storyplace.week-lot yearpeoplepeople worldplacestateweekpart country";
        patternMatch = "week.";
        patternNoMatch = "family";
        matches.add(40);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 34, was "+comparator.getComparisonCount(), 34 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_70() {
        if (skipGalil) return;
        text = "peoplefactwaterissue areaworldschooltime.childwater-systemhomewater world-state.life factplacewordfactplacenumber-worldnumberchildschool moneyfact";
        patternMatch = "place";
        patternNoMatch = "month";
        matches.add(89);
        matches.add(102);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 51, was "+comparator.getComparisonCount(), 51 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 36, was "+comparator.getComparisonCount(), 36 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_71() {
        if (skipGalil) return;
        text = "country.lottimelife-partgovernmentright groupissuechild daynumber issuestateprogram.company night-story.areahome.water-school night company-country ";
        patternMatch = "time";
        patternNoMatch = "way";
        matches.add(11);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 49, was "+comparator.getComparisonCount(), 49 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 54, was "+comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_72() {
        if (skipGalil) return;
        text = "monthnumberpeoplecasewordlot.roomschoolfact timenight numberworldthingstudysystemcompany.number-room-water waymoney issue.programrightpeoplerightlotissueprogramlotareaplacelotstudent bookchildnight stateyearthing companyfamilygovernmentschoolfactwaterday";
        patternMatch = "system";
        patternNoMatch = "home";
        matches.add(75);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 64, was "+comparator.getComparisonCount(), 64 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 91, was "+comparator.getComparisonCount(), 91 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_73() {
        if (skipGalil) return;
        text = "righthomeworldfact-";
        patternMatch = "fact-";
        patternNoMatch = "family";
        matches.add(14);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 13, was "+comparator.getComparisonCount(), 13 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 8, was "+comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_74() {
        if (skipGalil) return;
        text = "yearchildpeople monthweekfamily.waterpartpartwater familysystemcompanythingweekissuesystemhome week-life childhomerightstudyrightlotstudyarea";
        patternMatch = "system";
        patternNoMatch = "group";
        matches.add(57);
        matches.add(84);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 35, was "+comparator.getComparisonCount(), 35 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_75() {
        if (skipGalil) return;
        text = "daypart-governmentplacelot-fact";
        patternMatch = "fact";
        patternNoMatch = "thing";
        matches.add(27);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was "+comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 10, was "+comparator.getComparisonCount(), 10 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_76() {
        if (skipGalil) return;
        text = "moneysystem-countrystateworldnightcountryprogramstory.yeararea-programnumberissueplacemonthfamilygovernment-familygovernmentschoolgovernmentsystemlothomeareamonth systempartgroup-year";
        patternMatch = "government";
        patternNoMatch = "study";
        matches.add(97);
        matches.add(114);
        matches.add(130);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 66, was "+comparator.getComparisonCount(), 66 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 47, was "+comparator.getComparisonCount(), 47 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_77() {
        if (skipGalil) return;
        text = "studentnightfamilystudentwayissue-worldgrouphome-programarea placeday-story companypointlifethingcase.point.pointfamily-storyword timenight right-wordcasecompanyfamily-";
        patternMatch = "time";
        patternNoMatch = "money";
        matches.add(130);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 54, was "+comparator.getComparisonCount(), 54 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_78() {
        if (skipGalil) return;
        text = "state caseyearsystem dayroompeopletimewordschool wayworldwordcountryfamilybookchildthingnumberlifethingareacompanygroupchildcompany.people.childpointrightstory";
        patternMatch = "people";
        patternNoMatch = "government";
        matches.add(28);
        matches.add(132);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 48, was "+comparator.getComparisonCount(), 48 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 29, was "+comparator.getComparisonCount(), 29 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_79() {
        if (skipGalil) return;
        text = "familynight-people";
        patternMatch = "people";
        patternNoMatch = "program";
        matches.add(12);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 14, was "+comparator.getComparisonCount(), 14 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 8, was "+comparator.getComparisonCount(), 8 >= comparator.getComparisonCount());
    }

    @Test(timeout = TIMEOUT)
    public void testGalil_80() {
        if (skipGalil) return;
        text = "room";
        patternMatch = "room";
        patternNoMatch = "case";
        matches.add(0);
        comparator = new CharacterComparator();
        assertEquals(matches, PatternMatching.boyerMooreGalilRule(patternMatch, text, comparator));
        assertTrue("Comparison count should be 7, was "+comparator.getComparisonCount(), 7 >= comparator.getComparisonCount());
        comparator = new CharacterComparator();
        assertEquals(empty, PatternMatching.boyerMooreGalilRule(patternNoMatch, text, comparator));
        assertTrue("Comparison count should be 4, was "+comparator.getComparisonCount(), 4 >= comparator.getComparisonCount());
    }
}