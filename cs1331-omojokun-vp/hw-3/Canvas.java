//In order to help learn course concepts, I worked on the homework with Devam Shrivastava.
import java.util.Scanner;
public class Canvas {
    public static void main(String[] args) {
        int row = Integer.parseInt(args[0]);
        int col = Integer.parseInt(args[1]);
        CanvasRenderer.setup(row, col);
        int[][] canvas = new int[row][col];
        int color = getColor("#FFFFFF");
        fill(canvas, color);
        CanvasRenderer.render(canvas);
        Scanner scan = new Scanner(System.in);
        while (true) {
            System.out.print("Write your instruction. Press enter to update your canvas. ");
            String[] instructions = scan.nextLine().split(" ");
            if (instructions[0].equals("QUIT")) {
                break;
            } else if (instructions[0].equals("SAVE")) {
                CanvasImageSave.save(canvas, instructions[1]);
            } else {
                if (instructions[0].equals("PIXEL")) {
                    int row1 = Integer.parseInt(instructions[1]);
                    int col1 = Integer.parseInt(instructions[2]);
                    int color1 = getColor(instructions[3]);
                    setPixelColor(canvas, row1, col1, color1);
                } else if (instructions[0].equals("RECTANGLE")) {
                    int row2 = Integer.parseInt(instructions[1]);
                    int col2 = Integer.parseInt(instructions[2]);
                    int height2 = Integer.parseInt(instructions[3]);
                    int width2 = Integer.parseInt(instructions[4]);
                    int color2 = getColor(instructions[5]);
                    drawRectangle(canvas, row2, col2, height2, width2, color2);
                } else if (instructions[0].equals("SQUARE")) {
                    int row3 = Integer.parseInt(instructions[1]);
                    int col3 = Integer.parseInt(instructions[2]);
                    int size3 = Integer.parseInt(instructions[3]);
                    int color3 = getColor(instructions[4]);
                    drawSquare(canvas, row3, col3, size3, color3);
                } else if (instructions[0].equals("FILL")) {
                    int color4 = getColor(instructions[1]);
                    fill(canvas, color4);
                } else if (instructions[0].equals("REPLACE")) {
                    int color4 = getColor(instructions[1]);
                    int color5 = getColor(instructions[2]);
                    replaceColor(canvas, color4, color5);
                } else if (instructions[0].equals("GRID")) {
                    if (instructions[4].equals("SAME")) {
                        int row6 = Integer.parseInt(instructions[1]);
                        int col6 = Integer.parseInt(instructions[2]);
                        int size6 = Integer.parseInt(instructions[3]);
                        Boolean same6 = false;
                        int color6 = getColor(instructions[5]);
                        drawGrid(canvas, row6, col6, size6, same6, color6);
                    } else if (instructions[4].equals("DIFFERENT")) {
                        int row6 = Integer.parseInt(instructions[1]);
                        int col6 = Integer.parseInt(instructions[2]);
                        int size6 = Integer.parseInt(instructions[3]);
                        Boolean same6 = true;
                        int color6 = getColor(instructions[5]);
                        drawGrid(canvas, row6, col6, size6, same6, color6);
                    }
                } else if (instructions[0].equals("LOWER")) {
                    int row8 = Integer.parseInt(instructions[1]);
                    int col8 = Integer.parseInt(instructions[2]);
                    int size8 = Integer.parseInt(instructions[3]);
                    int color8 = getColor(instructions[4]);
                    drawLowerTriangle(canvas, row8, col8, size8, color8);
                } else if (instructions[0].equals("UPPER")) {
                    int row9 = Integer.parseInt(instructions[1]);
                    int col9 = Integer.parseInt(instructions[2]);
                    int size9 = Integer.parseInt(instructions[3]);
                    int color9 = getColor(instructions[4]);
                    drawUpperTriangle(canvas, row9, col9, size9, color9);
                }
                CanvasRenderer.render(canvas);
            }
        }
        scan.close();
        CanvasRenderer.close();
    }
    public static int getColor(String color) {
        int rgb = Integer.parseInt(color.substring(1), 16);
        return rgb;
    }
    public static void setPixelColor(int[][] canvas, int row, int col, int color) {
        canvas[row][col] = color;
    }
    public static void drawRectangle(int[][] canvas, int row, int col, int height, int width, int color) {
        for (int a = row; a < row + height; a++) {
            for (int b = col; b < col + width; b++) {
                canvas[a][b] = color;
            }
        }
    }
    public static void drawSquare(int[][] canvas, int row, int col, int size, int color) {
        drawRectangle(canvas, row, col, size, size, color);
    }
    public static void fill(int[][] canvas, int color) {
        for (int a = 0; a < canvas.length; a++) {
            drawRectangle(canvas, a, 0, 1, canvas[a].length, color);
        }
    }
    public static void replaceColor(int[][] canvas, int oldColor, int newColor) {
        for (int a = 0; a < canvas.length; a++) {
            for (int b = 0; b < canvas[a].length; b++) {
                if (canvas[a][b] == oldColor) {
                    canvas[a][b] = newColor;
                }
            }
        }
    }
    public static void drawGrid(int[][] canvas, int row, int col, int size, boolean same, int color) {
        for (int a = row; a < row + size; a++) {
            for (int b = col; b < col + size; b++) {
                if (((a % 2 == b % 2) && (same)) || ((a % 2 != b % 2) && (!same))) {
                    canvas[a][b] = color;
                }
            }
        }
    }
    public static void drawLowerTriangle(int[][] canvas, int row, int col, int size, int color) {
        for (int a = 0; a < size; a++) {
            for (int b = 0; b <= a; b++) {
                setPixelColor(canvas, row + a, col + b, color);
            }
        }
    }
    public static void drawUpperTriangle(int[][] canvas, int row, int col, int size, int color) {
        for (int a = 0; a < size; a++) {
            for (int b = a; b < size; b++) {
                setPixelColor(canvas, row + a, col + b, color);
            }
        }
    }
}
