//I worked on the assignment alone, using only course-provided materials.
import java.util.Scanner;
import java.text.DecimalFormat;
public class StudentCenter {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        DecimalFormat formatter = new DecimalFormat(".00");
        
        System.out.println("Welcome to the Student Center!");
        System.out.println("Activities\nBowling: $4.00 ($2 to rent bowling shoes)");
        System.out.println("Billiards: $5.00\nFood\nPizza: $8.50\nSalad: $7.00\n\n");
        
        System.out.print("What activity would you like to do? ");
        String activity = scan.nextLine();
        activity = activity.toLowerCase();
        double activityPrice = 0.00;
        if (activity.equals("bowling")) {
            activityPrice += 4.00;
            System.out.print("Do you need bowling shoes? ");
            String response = scan.nextLine();
            response = response.toLowerCase();
            if (response.equals("yes")) {
                activityPrice += 2.00;
            } else {
                activityPrice += 0.00;
            }
        } else {
            activityPrice += 5.00;
        }
        
        
        System.out.print("\n\nWhat food would you like? ");
        String food = scan.nextLine();
        food = food.toLowerCase();
        double foodPrice = 0.00;
        if (food.equals("pizza")) {
            foodPrice += 8.50;
            System.out.print("Choose a topping (mushrooms: $1.5, pepperoni: $1, none: $0) ");
            String topping = scan.nextLine();
            topping = topping.toLowerCase();
            switch (topping) {
                case "mushrooms":
                    foodPrice += 1.50;
                    break;
                case "pepperoni":
                    foodPrice += 1.00;
                    break;
                case "none":
                    break;
            }
        } else if (food.equals("salad")) {
            foodPrice += 7.00;
        }

        System.out.print("\n\nWhat percentage would you like to tip for the food? ");
        double tip = scan.nextDouble();
        if (tip < 0) {
            tip = 0.18;
        }

        System.out.print("\n\nHow many people are with you? ");
        int people = scan.nextInt();
        if (people < 0) {
            people = 0;
        }
 
        double subtotal = ((activityPrice + foodPrice) * (people + 1));
        double foodTip = ((foodPrice) * (people + 1) * (tip));
        double total = subtotal + foodTip;
        System.out.println("\n\nSubtotal: $" + formatter.format(subtotal));
        System.out.println("Food Tip: $" + formatter.format(foodTip));
        System.out.println("Total: $" + formatter.format(total));
    }
}
