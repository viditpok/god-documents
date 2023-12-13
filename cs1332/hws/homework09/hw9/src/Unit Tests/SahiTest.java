import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;

/**
 * @author Sahithya Pasagada
 * @version 1.0
 */
public class SahiTest {
    private static final int TIMEOUT = 200;
    private String kmpPattern;
    private String kmpText;
    private List<Integer> kmpAnswer;
    
    private String bmPattern;
    private String bmText;
    private List<Integer> bmAnswer;
    
    private String rkPattern;
    private String rkText;
    private List<Integer> rkAnswer;
    
    private CharacterComparator comparator;
    private List<Integer> emptyList = new ArrayList<>();


    @Before
    public void setUp() {
        comparator = new CharacterComparator();
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBuildFailureTableException1() {
    	kmpPattern = null;
    	int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBuildFailureTableException2() {
    	comparator = null;
    	int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKMPMatchException1() {
    	kmpPattern = null;
    	PatternMatching.kmp(kmpPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKMPMatchException2() {
    	kmpPattern = "";
    	PatternMatching.kmp(kmpPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKMPMatchException3() {
    	kmpText = null;
    	PatternMatching.kmp(kmpPattern, kmpText, comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testKMPMatchException4() {
    	comparator = null;
    	PatternMatching.kmp("abc", "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testLastTtableException() {
    	bmPattern = null;
    	Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(bmPattern);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBMMatchException1() {
    	bmPattern = null;
    	PatternMatching.boyerMoore(bmPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBMMatchException2() {
    	bmPattern = "";
    	PatternMatching.boyerMoore(bmPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBMMatchException3() {
    	bmText = null;
    	PatternMatching.boyerMoore(bmPattern, bmText, comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testBMMatchException4() {
    	comparator = null;
    	PatternMatching.boyerMoore("abc", "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRKMatchException1() {
    	rkPattern = null;
    	PatternMatching.rabinKarp(rkPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRKMatchException2() {
    	rkPattern = "";
    	PatternMatching.rabinKarp(rkPattern, "abc", comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRKMatchException3() {
    	rkText = null;
    	PatternMatching.rabinKarp(rkPattern, rkText, comparator);
    }
    
    @Test(timeout = TIMEOUT, expected = IllegalArgumentException.class)
    public void testRKMatchException4() {
    	comparator = null;
    	PatternMatching.rabinKarp("abc", "abc", comparator);
    }

    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable1() {
        /*
            pattern: abcdabeabf
            failure table: [0, 0, 0, 0, 1, 2, 0, 1, 2, 0]
            comparisons: 11
         */
    	kmpPattern = "abcdabeabf";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 0, 1, 2, 0, 1, 2, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable2() {
        /*
            pattern: abcdeabfabc
            failure table: [0, 0, 0, 0, 1, 2, 0, 1, 2, 3]
            comparisons: 11
         */
    	kmpPattern = "abcdeabfabc";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 3};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable3() {
        /*
            pattern: abcabc
            failure table: [0, 0, 0, 1, 2, 3]
            comparisons: 5
         */
    	kmpPattern = "abcabc";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 1, 2, 3};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 5.", 5, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable4() {
        /*
            pattern: aabcadaabe
            failure table: [0, 1, 0, 0, 1, 0, 1, 2, 3, 0]
            comparisons: 12
         */
    	kmpPattern = "aabcadaabe";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 1, 0, 0, 1, 0, 1, 2, 3, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 12.", 12, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable5() {
        /*
            pattern: aaaabaacd
            failure table: [0, 1, 2, 3, 0, 1, 2, 0, 0]
            comparisons: 13
         */
    	kmpPattern = "aaaabaacd";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 1, 2, 3, 0, 1, 2, 0, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 13.", 13, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable6() {
        /*
            pattern: aaaaaaaaaa
            failure table: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            comparisons: 9
         */
    	kmpPattern = "aaaaaaaaaa";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 9.", 9, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable7() {
        /*
            pattern: abcde
            failure table: [0, 0, 0, 0, 0]
            comparisons: 4
         */
    	kmpPattern = "abcde";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 0, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 4.", 4, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable8() {
        /*
            pattern: abcccccc
            failure table: [0, 0, 0, 0, 0, 0, 0, 0]
            comparisons: 7
         */
    	kmpPattern = "abcccccc";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 7.", 7, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable9() {
        /*
            pattern: abcdefbcdabcaab
            failure table: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 1, 2]
            comparisons: 16
         */
    	kmpPattern = "abcdefbcdabcaab";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 1, 2};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 16.", 16, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable10() {
        /*
            pattern: abababdefgh
            failure table: [0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0]
            comparisons: 12
         */
    	kmpPattern = "abababdefgh";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 12.", 12, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable12() {
        /*
            pattern: lslssslllslsll
            failure table: [0, 0, 1, 2, 0, 0, 1, 1, 1, 2, 3, 4, 3, 1]
            comparisons: 19
         */
    	kmpPattern = "lslssslllslsll";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 1, 2, 0, 0, 1, 1, 1, 2, 3, 4, 3, 1};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 19.", 19, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTable11() {
        /*
            pattern: xyxyyxyxyxyyz
            failure table: [0, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4, 5, 0]
            comparisons: 15
         */
    	kmpPattern = "xyxyyxyxyxyyz";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4, 5, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 15.", 15, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableSpecialCharacters() {
        /*
            pattern: !#$!!!^!
            failure table: [0, 0, 0, 1, 1, 1, 0, 1]
            comparisons: 10
         */
    	kmpPattern = "!#$!!!^!";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 0, 0, 1, 1, 1, 0, 1};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 10.", 10, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableWithSpaces() {
        /*
            pattern: _____a
            failure table: [0, 1, 2, 3, 4, 0]
            comparisons: 10
         */
    	kmpPattern = "     a";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {0, 1, 2, 3, 4, 0};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 9.", 9, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildFailureTableEmpty() {

    	kmpPattern = "";
        int[] failureTable = PatternMatching
                .buildFailureTable(kmpPattern, comparator);
        int[] expected = {};
        assertArrayEquals(expected, failureTable);
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() == 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchEndOnly() {
        /*
            pattern: abcde
            text: abababcabcde
            indices: 7
            expected total comparison: 15

            failure table: [0, 0, 0, 0, 0]
            comparisons: 4

         */
    	kmpPattern = "abcde";
    	kmpText = "abababcabcde";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(7);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 19.", 19, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchBeginningOnly() {
        /*
            pattern: ab
            text: abakafcajcde
            indices: 0
            expected total comparison: 15

            failure table: [0, 0]
            comparisons: 1

         */
    	kmpPattern = "ab";
    	kmpText = "abakafcajcde";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(0);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 15.", 15, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchMiddleOnly() {
        /*
            pattern: afc
            text: abakafcajcde
            indices: 4
            expected total comparison: 15

            failure table: [0, 0, 0]
            comparisons: 2

         */
    	kmpPattern = "afc";
    	kmpText = "abakafcajcde";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(4);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 15.", 15, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchNoMatch1() {
        /*
            pattern: abababa
            text: abcabaabaabaabc
            indices: []
            expected total comparison: 20

            failure table: [0, 0, 1, 2, 3, 4, 5]
            comparisons: 6

         */
    	kmpPattern = "abababa";
    	kmpText = "abcabaabaabaabc";
    	kmpAnswer = new ArrayList<>();
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 20.", 20, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchNoMatch2() {
        /*
            pattern: c
            text: a
            indices: []
            expected total comparison: 1

            failure table: [0]
            comparisons: 0

         */
    	kmpPattern = "c";
    	kmpText = "a";
    	kmpAnswer = new ArrayList<>();
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 1.", 1, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchNoMatch3() {
        /*
            pattern: defghi
            text: cbdabcdefghdefgh
            indices: []
            expected total comparison: 18

            failure table: [0, 0, 0, 0, 0]
            comparisons: 5

         */
    	kmpPattern = "defghi";
    	kmpText = "cbdabcdefghdefgh";
    	kmpAnswer = new ArrayList<>();
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 18.", 18, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchNoMatch4() {
        /*
            pattern: defghdhd
            text: cbdabcdefghdefgh
            indices: []
            expected total comparison: 22

            failure table: [0, 0, 0, 0, 0, 1, 0, 1]
            comparisons: 8

         */
    	kmpPattern = "defghdhd";
    	kmpText = "cbdabcdefghdefgh";
    	kmpAnswer = new ArrayList<>();
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 22.", 22, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatchNoMatch5() {
        /*
            pattern: a!!!a!!!a
            text: abca!cabcd
            indices: []
            expected total comparison: 11

            failure table: [0, 0, 0, 0, 1, 2, 3, 4, 5]
            comparisons: 8

         */
    	kmpPattern = "a!!!a!!!a";
    	kmpText = "abca!cabcd";
    	kmpAnswer = new ArrayList<>();
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    

    @Test(timeout = TIMEOUT)
    public void testKMPMatch1() {
        /*
            pattern: abcabc
            text:  abcabcabcabc
            indices: [0]
            expected total comparison: 17

            failure table: [0, 0, 0, 1, 2, 3]
            comparisons: 5

         */
    	kmpPattern = "abcabc";
    	kmpText = "abcabcabcabc";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(0);
    	kmpAnswer.add(3);
    	kmpAnswer.add(6);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch2() {
        /*
            pattern: abcabc
            text:  abcabcabcabc
            indices: [0, 3, 6]
            expected total comparison: 17

            failure table: [0, 0, 0, 1, 2, 3]
            comparisons: 5

         */
    	kmpPattern = "abcabc";
    	kmpText = "abcabcabcabc";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(0);
    	kmpAnswer.add(3);
    	kmpAnswer.add(6);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch3() {
        /*
            pattern: bab
            text:  abcdefghabab
            indices: [9]
            expected total comparison: 15

            failure table: [0, 0, 1]
            comparisons: 2

         */
    	kmpPattern = "bab";
    	kmpText = "abcdefghabab";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(9);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 15.", 15, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch4() {
        /*
            pattern: babbbab
            text:  babbbabacgfhibabbbabgf
            indices: [0, 13]
            expected total comparison: 30

            failure table: [0, 0, 1, 1, 1, 2, 3]
            comparisons: 8

         */
    	kmpPattern = "babbbab";
    	kmpText = "babbbabacgfhibabbbabgf";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(0);
    	kmpAnswer.add(13);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 30.", 30, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch5() {
        /*
            pattern: bbb
            text:  defbbbwqbbbad
            indices: [3, 8]
            expected total comparison: 17

            failure table: [0, 1, 2]
            comparisons: 2

         */
    	kmpPattern = "bbb";
    	kmpText = "defbbbwqbbbad";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(3);
    	kmpAnswer.add(8);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch6() {
        /*
            pattern: a__a
            text: fghfgha_a_a__abhg
            indices: [10]
            expected total comparison: 20

            failure table: [0, 0, 0, 1]
            comparisons: 3

         */
    	kmpPattern = "a  a";
    	kmpText = "fghfgha a a  abhg";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(10);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 20.", 20, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch7() {
        /*
            pattern: babbabab
            text: babbabab
            indices: [0]
            expected total comparison: 17

            failure table: [0, 0, 1, 1, 2, 3, 2, 3]
            comparisons: 9

         */
    	kmpPattern = "babbabab";
    	kmpText = "babbabab";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(0);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 17.", 17, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch8() {
        /*
            pattern: ababa
            text: ababbababaabababb
            indices: [5, 10]
            expected total comparison: 25

            failure table: [0, 0, 1, 2, 3]
            comparisons: 4

         */
    	kmpPattern = "ababa";
    	kmpText = "ababbababaabababb";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(5);
    	kmpAnswer.add(10);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 25.", 25, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPMatch9() {
        /*
            pattern: sossos
            text: osossosossosososossos
            indices: [1, 15]
            expected total comparison: 31

            failure table: [0, 0, 1, 1, 2, 3]
            comparisons: 6

         */
    	kmpPattern = "sossos";
    	kmpText = "osossosossosososossos";
    	kmpAnswer = new ArrayList<>();
    	kmpAnswer.add(1);
    	kmpAnswer.add(6);
    	kmpAnswer.add(15);
    	
        assertEquals(kmpAnswer,
                PatternMatching.kmp(kmpPattern, kmpText, comparator));
        assertTrue("Did not use the comparator.",
                comparator.getComparisonCount() != 0);
        assertEquals("Comparison count was " + comparator.getComparisonCount()
                + ". Should be 31.", 31, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildLastTable1() {
        /*
            pattern: xpbctbxabpqxctbpq
            last table: {x : 11, p : 15, b : 14, c : 12, t : 13, a : 7, q : 16}
         */
    	
    	bmPattern = "xpbctbxabpqxctbpq";
    	char[] keys = "xpbctaq".toCharArray();
    	int[] values = {11, 15, 14, 12, 13, 7, 16};
    	
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(bmPattern);
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        for (int i = 0; i < values.length; i++) {
        	expectedLastTable.put(keys[i], values[i]);
        }
        assertEquals(expectedLastTable, lastTable);
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildLastTable2() {
        /*
            pattern: bbbbbbbbbbb
            last table: {b: 10}
         */
    	
    	bmPattern = "bbbbbbbbbbb";
    	char[] keys = "b".toCharArray();
    	int[] values = {10};
    	
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(bmPattern);
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        for (int i = 0; i < values.length; i++) {
        	expectedLastTable.put(keys[i], values[i]);
        }
        
        Integer a = -1;
        assertEquals(lastTable.getOrDefault('a', -1), a);
        assertEquals(expectedLastTable, lastTable);
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildLastTable3() {
        /*
            pattern: _
            last table: { : 0}
         */
    	
    	bmPattern = " ";
    	char[] keys = " ".toCharArray();
    	int[] values = {0};
    	
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(bmPattern);
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        for (int i = 0; i < values.length; i++) {
        	expectedLastTable.put(keys[i], values[i]);
        }
        assertEquals(expectedLastTable, lastTable);
    }
    
    @Test(timeout = TIMEOUT)
    public void testBuildLastTable4() {
        /*
            pattern: qcabdabdab
            last table: {q : 0, c : 1, a : 8, b : 9, d : 7}
         */
    	
    	bmPattern = "qcabdabdab";
    	char[] keys = "qcabd".toCharArray();
    	int[] values = {0, 1, 8, 9, 7};
    	
        Map<Character, Integer> lastTable = PatternMatching
                .buildLastTable(bmPattern);
        Map<Character, Integer> expectedLastTable = new HashMap<>();
        for (int i = 0; i < values.length; i++) {
        	expectedLastTable.put(keys[i], values[i]);
        }
        assertEquals(expectedLastTable, lastTable);
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMooreCharNotInPatternNoMatch() {
    	/*
        pattern: sgdhss
        text: qewrtwyeueywueyuwu
        indices: -
        expected total comparisons: 3
    	 */
    	bmPattern = "sgdhss";
    	bmText = "qewrtwyeueywueyuwu";

    	assertEquals(emptyList, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 3.", 3, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMooreCharBeforeNoMatch() {
    	/*
        pattern: eager
        text: monkeyseverywhere
        indices: -
        expected total comparisons: 6
    	 */
    	bmPattern = "eager";
    	bmText = "monkeyseverywhere";

    	assertEquals(emptyList, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 6.", 6, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMooreCharAfterNoMatch() {
    	/*
        pattern: isomorphic
        text: homeomorphic
        indices: -
        expected total comparisons: 10
    	 */
    	bmPattern = "isomorphic";
    	bmText = "homeomorphic";

    	assertEquals(emptyList, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 10.", 10, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore1() {
       	/*
        pattern: derl
        text: wonderlandcrazy
        indices: [3]
        expected total comparisons: 8
    	 */
    	bmPattern = "derl";
    	bmText = "wonderlandcrazy";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(3);
    	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 8.", 8, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore2() {
       	/*
        pattern: ward
        text: forwardbackward
        indices: [3, 11]
        expected total comparisons: 11
    	 */
    	bmPattern = "ward";
    	bmText = "forwardbackward";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(3);
    	bmAnswer.add(11);
    	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore3() {
       	/*
        pattern: AGCGAG
        text: ATCTTAGAGCGAG
        indices: [7]
        expected total comparisons: 12
    	 */
    	bmPattern = "AGCGAG";
    	bmText = "ATCTTAGAGCGAG";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(7);
   	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 12.", 12, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore4() {
       	/*
        pattern: 1000
        text: 0000000000000000
        indices: -
        expected total comparisons: 52
    	 */
    	bmPattern = "1000";
    	bmText = "0000000000000000";
    	bmAnswer = new ArrayList<>();

    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 52.", 52, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore5() {
       	/*
        pattern: 1000
        text: 1010000100011000
        indices: [2, 7, 12]
        expected total comparisons: 23
    	 */
    	bmPattern = "1000";
    	bmText = "1010000100011000";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(2);
    	bmAnswer.add(7);
    	bmAnswer.add(12);

    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 23.", 23, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore6() {
       	/*
        pattern: abcabc
        text: abcabcxxxxxxx
        indices: [0]
        expected total comparisons: 8
    	 */
    	bmPattern = "abcabc";
    	bmText = "abcabcxxxxxxx";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(0);

    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 8.", 8, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore7() {
       	/*
        pattern: aaxaa
        text: aaxaaxaa
        indices: [0, 3]
        expected total comparisons: 11
    	 */
    	bmPattern = "aaxaa";
    	bmText = "aaxaaxaa";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(0);
    	bmAnswer.add(3);
    	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore8() {
       	/*
        pattern: aa
        text: abaaaaabaaaaaab
        indices: [2, 3, 4, 5, 8, 9, 10, 11, 12]
        expected total comparisons: 21
    	 */
    	bmPattern = "aa";
    	bmText = "abaaaaabaaaaaab";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(2);
    	bmAnswer.add(3);
    	bmAnswer.add(4);
    	bmAnswer.add(5);
    	bmAnswer.add(8);
    	bmAnswer.add(9);
    	bmAnswer.add(10);
    	bmAnswer.add(11);
    	bmAnswer.add(12);
    	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 21.", 21, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore9() {
       	/*
        pattern: ssss
        text: asssssssssa
        indices: [1, 2, 3, 4, 5, 6]
        expected total comparisons: 29
    	 */
    	bmPattern = "ssss";
    	bmText = "asssssssssa";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(1);
    	bmAnswer.add(2);
    	bmAnswer.add(3);
    	bmAnswer.add(4);
    	bmAnswer.add(5);
    	bmAnswer.add(6);

    	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 29.", 29, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMoore10() {
       	/*
        pattern: ys
        text: sassysassysweet
        indices: [4, 9]
        expected total comparisons: 14
    	 */
    	bmPattern = "ys";
    	bmText = "sassysassysweet";
    	bmAnswer = new ArrayList<>();
    	bmAnswer.add(4);
    	bmAnswer.add(9);

 	
    	assertEquals(bmAnswer, PatternMatching.boyerMoore(bmPattern,
                    bmText, comparator));
    	assertTrue("Did not use the comparator.",
    			comparator.getComparisonCount() != 0);
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 14.", 14, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch1() {
    	/*
        pattern: lemon
        text: freshjuice
        indices: 
        expected total comparisons: 0
     */
    	rkPattern = "lemon";
    	rkText = "freshjuice";
    	rkAnswer = new ArrayList<Integer>();
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch2() {
    	/*
        pattern: 11    1 1
        text: 1 1 1 1 1
        indices: 
        expected total comparisons: 0
     */
    	rkPattern = "1   1 1 1 1 1";
    	rkText = "1 1 1 1 1 1 1 1 1 1";
    	rkAnswer = new ArrayList<Integer>();
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch3() {
    	/*
        pattern: friend
        text: hello
        indices: 
        expected total comparisons: 0
    	*/
    	rkPattern = "friend";
    	rkText = "hello";
    	rkAnswer = new ArrayList<Integer>();
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch4() {
    	/*
        pattern: ababab
        text: h
        indices: 
        expected total comparisons: 0
    	*/
    	rkPattern = "ababab";
    	rkText = "h";
    	rkAnswer = new ArrayList<Integer>();
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpNoMatch5() {
    	/*
        pattern: morris
        text: boyermoorerabinkarp
        indices: 
        expected total comparisons: 0
    	*/
    	rkPattern = "morris";
    	rkText = "boyermoorerabinkarp";
    	rkAnswer = new ArrayList<Integer>();
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 0.", 0, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp1() {
    	/*
        pattern: nana
        text: bananaappleanna
        indices: 2
        expected total comparisons: 4
    	*/
    	rkPattern = "nana";
    	rkText = "bananaappleanna";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(2);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 4.", 4, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp2() {
    	/*
        pattern: cdd
        text: abccddaefg
        indices: 3
        expected total comparisons: 3
    	*/
    	rkPattern = "cdd";
    	rkText = "abccddaefg";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(3);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 3.", 3, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp3() {
    	/*
        pattern: aa
        text: aaaaaaaaaa
        indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected total comparisons: 18
    	*/
    	rkPattern = "aa";
    	rkText = "aaaaaaaaaa";
    	rkAnswer = new ArrayList<Integer>();
    	for (int i = 0; i <= 8; i++) {
    		rkAnswer.add(i);
    	}
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 18.", 18, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp4() {
    	/*
        pattern: abcd
        text: abcdbacdabcdeabcd
        indices: [0, 8, 13]
        expected total comparisons: 12
    	*/
    	rkPattern = "abcd";
    	rkText = "abcdbacdabcdeabcd";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(0);
    	rkAnswer.add(8);
    	rkAnswer.add(13);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 12.", 12, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp5() {
    	/*
        pattern: o
        text: helloworld
        indices: [4, 6]
        expected total comparisons: 2
    	*/
    	rkPattern = "o";
    	rkText = "helloworld";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(4);
    	rkAnswer.add(6);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 2.", 2, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp6() {
    	/*
        pattern: abracadabra
        text: hocuspocusabracadabra
        indices: [10]
        expected total comparisons: 11
    	*/
    	rkPattern = "abracadabra";
    	rkText = "hocuspocusabracadabra";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(10);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 11.", 11, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp7() {
    	/*
        pattern: bab
        text: ababababababa
        indices: [1, 3, 5, 7, 9]
        expected total comparisons: 15
    	*/
    	rkPattern = "bab";
    	rkText = "ababababababa";
    	rkAnswer = new ArrayList<Integer>();
    	for(int i = 1; i < 10; i+=2) {
    		rkAnswer.add(i);
    	}
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 15.", 15, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp8() {
    	/*
        pattern: a
        text: a
        indices: [0]
        expected total comparisons: 1
    	*/
    	rkPattern = "a";
    	rkText = "a";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(0);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 1.", 1, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp9() {
    	/*
        pattern: u
        text: sillydusk
        indices: [6]
        expected total comparisons: 1
    	*/
    	rkPattern = "u";
    	rkText = "sillydusk";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(6);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 1.", 1, comparator.getComparisonCount());
    }
    
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp10() {
    	/*
        pattern: ()
        text: ()()()()()((((()))))
        indices: [0, 2, 4, 6, 8, 14]
        expected total comparisons: 12
    	*/
    	rkPattern = "()";
    	rkText = "()()()()()((((()))))";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(0);
    	rkAnswer.add(2);
    	rkAnswer.add(4);
    	rkAnswer.add(6);
    	rkAnswer.add(8);
    	rkAnswer.add(14);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 12.", 12, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp11() {
    	/*
        pattern: ♥♥
        text: ♥♥♥♥
        indices: [0, 1, 2]
        expected total comparisons: 6
    	*/
    	rkPattern = "♥♥";
    	rkText = "♥♥♥♥";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(0);
    	rkAnswer.add(1);
    	rkAnswer.add(2);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 6.", 6, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp12() {
    	/*
        pattern: ily
        text: funnilycrepilysleepily
        indices: [4, 11, 19]
        expected total comparisons: 9
    	*/
    	rkPattern = "ily";
    	rkText = "funnilycrepilysleepily";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(4);
    	rkAnswer.add(11);
    	rkAnswer.add(19);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 9.", 9, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarp13() {
    	/*
        pattern: fghsgshsh
        text: fghsgshsh
        indices: [0]
        expected total comparisons: 9
    	*/
    	rkPattern = "fghsgshsh";
    	rkText = "fghsgshsh";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(0);

    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 9.", 9, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpSpecialCharacters() {
    	/*
        pattern: ^&
        text: !@#^&&^&^&_
        indices: [3, 6, 8]
        expected total comparisons: 6
     */
    	rkPattern = "^&";
    	rkText = "!@#^&&^&^& ";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(3);
    	rkAnswer.add(6);
    	rkAnswer.add(8);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 6.", 6, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testKMPLongMatch() {
    	/*
        pattern: jfh
        text: adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf
        indices: [10, 20, 31, 43, 87, 109]
        expected total comparisons: 123
    	*/
    	kmpPattern = "jfh";
    	kmpText = "adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf";
    	kmpAnswer = new ArrayList<Integer>();
    	kmpAnswer.add(10);
    	kmpAnswer.add(20);
    	kmpAnswer.add(31);
    	kmpAnswer.add(43);
    	kmpAnswer.add(87);
    	kmpAnswer.add(109);
    	
    	assertEquals(kmpAnswer,
    			PatternMatching.kmp(kmpPattern, kmpText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 123.", 123, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testRabinKarpLongMatch() {
    	/*
        pattern: jfh
        text: adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf
        indices: [10, 20, 31, 43, 87, 109]
        expected total comparisons: 18
    	*/
    	rkPattern = "jfh";
    	rkText = "adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf";
    	rkAnswer = new ArrayList<Integer>();
    	rkAnswer.add(10);
    	rkAnswer.add(20);
    	rkAnswer.add(31);
    	rkAnswer.add(43);
    	rkAnswer.add(87);
    	rkAnswer.add(109);
    	
    	assertEquals(rkAnswer,
    			PatternMatching.rabinKarp(rkPattern, rkText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 18.", 18, comparator.getComparisonCount());
    }
    
    @Test(timeout = TIMEOUT)
    public void testBoyerMooreLongMatch() {
    	/*
        pattern: jfh
        text: adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf
        indices: [10, 20, 31, 43, 87, 109]
        expected total comparisons: 71
    	*/
    	bmPattern = "jfh";
    	bmText = "adhskjdhskjfhdsfhsdkjfhdskfhdskjfhdsjkfhdskjfhuejhdkfhdsjkfhjksdhfjksdhfjkdshfdsjkfhdskjfhdseuihdsbsdbvnmnbdkjfhekf";
    	bmAnswer = new ArrayList<Integer>();
    	bmAnswer.add(10);
    	bmAnswer.add(20);
    	bmAnswer.add(31);
    	bmAnswer.add(43);
    	bmAnswer.add(87);
    	bmAnswer.add(109);
    	
    	assertEquals(bmAnswer,
    			PatternMatching.boyerMoore(bmPattern, bmText, comparator));
    	assertEquals("Comparison count was " + comparator.getComparisonCount()
            	+ ". Should be 71.", 71, comparator.getComparisonCount());
    }
}
