/*In order to help learn course concepts, I
consulted related material that can be found at
https://docs.oracle.com/javase/8/javafx/api/javafx/scene/layout/GridPane.html
*/
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.stage.Stage;
import javafx.scene.paint.ImagePattern;
import javafx.scene.shape.Rectangle;
import javafx.scene.layout.GridPane;


/**
 * @author Vidit Pokharna
 * @version 1.0
 */
public class StarterUpper extends Application {
    /**
     * @param s stage s
     * @throws IOException exception thrown
     */
    public void start(Stage s) throws IOException {
        List<StartUpIdea> ideas = new ArrayList<StartUpIdea>();
        File newFile = new File("ideas.txt");
        newFile.createNewFile();

        s.setTitle("Problem Ideation Form.");
        GridPane r = new GridPane();
        r.setPadding(new Insets(5, 5, 5, 5));
        r.setVgap(20);
        r.setHgap(5);

        Label l = new Label("Vidit Pokharna");
        r.addRow(0, l);
        Label q1 = new Label("What is the problem?");
        TextField t1 = new TextField();
        r.addRow(1, q1, t1);
        Label q2 = new Label("Who is the target customer?");
        TextField t2 = new TextField();
        r.addRow(2, q2, t2);
        Label q3 = new Label("How badly does the customer NEED this problem fixed (1-10)?");
        TextField t3 = new TextField();
        r.addRow(3, q3, t3);
        Label q4 = new Label("How many people do you know who might experience this problem?");
        TextField t4 = new TextField();
        r.addRow(4, q4, t4);
        Label q5 = new Label("How big is the target market?");
        TextField t5 = new TextField();
        r.addRow(5, q5, t5);
        Label q6 = new Label("Who are the competitors/existing solutions?");
        TextField t6 = new TextField();
        r.addRow(6, q6, t6);
        Label q7 = new Label("How long has this problem existed, to your best estimate?");
        TextField t7 = new TextField();
        r.addRow(7, q7, t7);

        //Button to save the start-up idea
        //Contains lambda function
        Button b1 = new Button();
        b1.setText("Save Idea");
        b1.setOnAction(event -> {
                if (t1.getText() == null
                    || t2.getText() == null
                    || t3.getText() == null
                    || t4.getText() == null
                    || t5.getText() == null
                    || t6.getText() == null
                    || t7.getText() == null) {
                    Alert a = new Alert(Alert.AlertType.ERROR);
                    a.setTitle("ERROR");
                    a.setHeaderText("One or more of the inputs are empty");
                    a.show();
                } else if (Integer.parseInt(t3.getText()) > 10
                    || Integer.parseInt(t3.getText()) < 1
                    || Integer.parseInt(t4.getText()) < 0
                    || Integer.parseInt(t5.getText()) < 0
                    || Integer.parseInt(t7.getText()) < 0) {
                    Alert a = new Alert(Alert.AlertType.ERROR);
                    a.setTitle("ERROR");
                    a.setHeaderText("One or more of the inputs are out of bounds");
                    a.show();
                } else if (!t1.getText().matches("[a-zA-Z0-9 ]+")
                    || !t2.getText().matches("[a-zA-Z0-9 ]+")
                    || !t3.getText().matches("\\d+")
                    || !t4.getText().matches("^[0-9]*[1-9][0-9]*$")
                    || !t5.getText().matches("^[0-9]*[1-9][0-9]*$")
                    || !t6.getText().matches("[a-zA-Z0-9 ]+")
                    || !t7.getText().matches("^[0-9]*[1-9][0-9]*$")) {
                    Alert a = new Alert(Alert.AlertType.ERROR);
                    a.setTitle("ERROR");
                    a.setHeaderText("One or more of the inputs do not have the correct format");
                    a.show();
                } else {
                    ideas.add(new StartUpIdea(t1.getText(), t2.getText(), Integer.parseInt(t3.getText()),
                        Integer.parseInt(t4.getText()), Integer.parseInt(t5.getText()), t6.getText(),
                        Integer.parseInt(t7.getText())));
                    t1.clear();
                    t2.clear();
                    t3.clear();
                    t4.clear();
                    t5.clear();
                    t6.clear();
                    t7.clear();
                }
            });

        //Button for sorting through the saved idea
        //Contains the anonymous inner class
        Button b2 = new Button();
        b2.setText("Sort Ideas");
        EventHandler<ActionEvent> eh = new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent ae) {
                Collections.sort(ideas);
            }
        };
        b2.setOnAction(eh);

        //Button for saving the current list of ideas to a file
        //Contains lambda function
        Button b3 = new Button();
        b3.setText("Save Ideas to File");
        b3.setOnAction((event) -> {
                if (!newFile.exists()) {
                    try {
                        newFile.createNewFile();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                FileUtil.saveIdeasToFile(ideas, newFile);
            });

        Button b4 = new Button();
        b4.setText("Reset");
        EventHandler<ActionEvent> resetEvent = new EventHandler<ActionEvent>() {
            public void handle(ActionEvent arg0) {
                Alert a = new Alert(Alert.AlertType.CONFIRMATION);
                a.setTitle("WARNING");
                a.setHeaderText("Are you sure you want to reset?");
                Optional<ButtonType> result = a.showAndWait();
                if (result.get().equals(ButtonType.OK)) {
                    if (newFile.exists()) {
                        newFile.delete();
                        ideas.clear();
                        t1.clear();
                        t2.clear();
                        t3.clear();
                        t4.clear();
                        t5.clear();
                        t6.clear();
                        t7.clear();
                    }
                    ideas.clear();
                    t1.clear();
                    t2.clear();
                    t3.clear();
                    t4.clear();
                    t5.clear();
                    t6.clear();
                    t7.clear();
                } else if (result.get().equals(ButtonType.NO)) {
                    a.close();
                }
            }
        };
        b4.setOnAction(resetEvent);
        r.addRow(8, b1, b2, b3, b4);

        Rectangle rec = new Rectangle();
        rec.setX(30);
        rec.setY(30);
        rec.setWidth(300);
        rec.setHeight(100);
        rec.setArcWidth(20);
        rec.setArcHeight(20);
        Image image1 = new Image("https://picsum.photos/500/300?random=1.png");
        rec.setFill(new ImagePattern(image1));
        r.addRow(9, rec);

        r.setStyle("-fx-background-color: #FFCCCB;");

        Scene scene = new Scene(r, 850, 600);
        s.setScene(scene);
        s.show();
    }

    /**
     * @param args string[]
     */
    public static void main(String[] args) {
        launch(args);
    }
}