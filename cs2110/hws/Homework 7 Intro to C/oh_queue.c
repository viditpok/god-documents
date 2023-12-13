// DO NOT MODIFY THE INCLUDE(S) LIST
#include <stdio.h>
#include "oh_queue.h"

struct Queue oh_queue;

/** push
 * @brief Create a new student and push him
 * onto the OH queue
 * @param studentName pointer to the student's name
 * @param topicName topic the student has a question on
 * @param questionNumber hw question number student has a question on
 * @param pub_key public key used for calculating the hash for customID
 * @return FAILURE if the queue is already at max length, SUCCESS otherwise
 */
int push(const char *studentName, const enum subject topicName, const float questionNumber, struct public_key pub_key){
    if (studentName == NULL || oh_queue.stats.no_of_people_in_queue == MAX_QUEUE_LENGTH) {
       return FAILURE;
   }

    struct Topic topic;
    struct StudentData newStudentData;
    struct Student newStudent;

    topic.topicName = topicName;
    topic.questionNumber = questionNumber;
    newStudentData.topic = topic;

    size_t studentNameLen = my_strlen(studentName);
    
    if (studentNameLen >= MAX_NAME_LENGTH) {
        newStudentData.name[MAX_NAME_LENGTH - 1] = 0;
        for (size_t i = 0; i < MAX_NAME_LENGTH - 1; i++) {
            newStudentData.name[i] = *studentName;
            studentName++;
        }
    } else {
        for (size_t i = 0; i <= studentNameLen; i++) {
            newStudentData.name[i] = *studentName;
            studentName++;
        }
    }
    
    newStudent.studentData = newStudentData;
    hash(newStudent.customID, newStudentData.name, pub_key);
    newStudent.queue_number = oh_queue.stats.no_of_people_visited + oh_queue.stats.no_of_people_in_queue;


    oh_queue.students[oh_queue.stats.no_of_people_in_queue++] = newStudent;
    OfficeHoursStatus(&oh_queue.stats);

    return SUCCESS;
}



/** pop
 * @brief Pop a student out the OH queue
 * @return FAILURE if the queue is already at empty, SUCCESS otherwise
 */
int pop(void) {
    if (oh_queue.stats.no_of_people_in_queue == 0) {
        oh_queue.stats.currentStatus = "Completed";
        return FAILURE;
    }
    
    struct Student *current_student = oh_queue.students;
    struct Student *next_student = oh_queue.students + 1;

    while (next_student < oh_queue.students + oh_queue.stats.no_of_people_in_queue) {
        *current_student = *next_student;
        current_student++;
        next_student++;
    }
    
    oh_queue.stats.no_of_people_in_queue--;
    oh_queue.stats.no_of_people_visited++;
    OfficeHoursStatus(&oh_queue.stats);

    return SUCCESS;
}

/** group_by_topic
 * @brief Push pointers to students, who match the given topic, to
 * the given array "grouped"
 * @param topic the topic the students need to match
 * @param grouped an array of pointers to students
 * @return the number of students matched
 */
int group_by_topic(struct Topic topic, struct Student *grouped[]) { 
    int count = 0;
    int size = oh_queue.stats.no_of_people_in_queue;

    for (int a = 0; a < size; a++) {
        if (topic.topicName == oh_queue.students[a].studentData.topic.topicName) {
            if (topic.questionNumber == oh_queue.students[a].studentData.topic.questionNumber) {
                grouped[count] = &oh_queue.students[a];
                count++;
            }
        }
    }

    return count; 
}

/** hash
 * @brief Creates a hash based on pub_key provided
 * @param ciphertext the pointer where you will store the hashed text
 * @param plaintext the original text you need to hash
 * @param pub_key public key used for calculating the hash
 */
void hash(int *ciphertext, char *plaintext, struct public_key pub_key) {
    while (*plaintext != '\0') {
        int c = power_and_mod((int) *plaintext, pub_key.e, pub_key.n);
        *ciphertext = c;
        ciphertext++;
        plaintext++;
    }
    *ciphertext = '\0';
}

/** update_student
 * @brief Find the student with the given ID and update his topic
 * @param customID a pointer to the id of the student you are trying to find
 * @param newTopic the new topic that should be assigned to him
 * @return FAILURE if no student is matched, SUCCESS otherwise
 */
int update_student(struct Topic newTopic, int *customID) {
    for (int a = 0; a < oh_queue.stats.no_of_people_in_queue; a++) {
        if (*oh_queue.students[a].customID == *customID) {
            oh_queue.students[a].studentData.topic.topicName = newTopic.topicName;
            oh_queue.students[a].studentData.topic.questionNumber = newTopic.questionNumber;
            return SUCCESS;
        }
    }

    return FAILURE;
}

/** remove_student_by_name
 * @brief Removes first instance of a student with the given name
 * @param name the name you are searching for
 * @return FAILURE if no student is matched, SUCCESS otherwise
 */
int remove_student_by_name(char *name){
    int size = oh_queue.stats.no_of_people_in_queue;
    for (int a = 0; a < size; a++) {
        if (my_strncmp(oh_queue.students[a].studentData.name, name, MAX_NAME_LENGTH) == 0) {
            for (int b = a; b < size - 1; b++) {
                oh_queue.students[b] = oh_queue.students[b + 1];
            }
            oh_queue.stats.no_of_people_in_queue--;
            oh_queue.stats.no_of_people_visited++;
            OfficeHoursStatus(&oh_queue.stats);
            return SUCCESS;
        }
    }
    return FAILURE;
}

/** remove_student_by_topic
 * @brief Remove all instances of students with the given topic
 * @param topic the topic you are trying to remove from the queue
 * @return FAILURE if no student is matched, SUCCESS otherwise
 */
int remove_student_by_topic(struct Topic topic) {
    int size = oh_queue.stats.no_of_people_in_queue;
    int numRemoved = 0;
    for (int a = 0; a < size; a++) {
        if (topic.topicName == oh_queue.students[a].studentData.topic.topicName) {
            if (topic.questionNumber == oh_queue.students[a].studentData.topic.questionNumber) {
                numRemoved++;
                for (int b = a; b < size - 1; b++) {
                    oh_queue.students[b] = oh_queue.students[b + 1];
                }
                oh_queue.stats.no_of_people_in_queue--;
                oh_queue.stats.no_of_people_visited++;
                a--;
                size--;
            }
        }
    }
    if (numRemoved > 0) {
        OfficeHoursStatus(&oh_queue.stats);
        return SUCCESS;
    } else {
        return FAILURE;
    }
}

/** OfficeHoursStatus
 * @brief Updates the "currentStatus" field based on
 * whether or not all students in the queue have been helped
 * @param resultStats A pointer the OfficeHoursStats variable
 * you are to update
 */
void OfficeHoursStatus(struct OfficeHoursStats* resultStats){
    if (oh_queue.stats.no_of_people_in_queue == 0) {
        resultStats->currentStatus = "Completed";
    } else {
        resultStats->currentStatus = "InProgress";
    }
}

/*
 * Calculates (b^e)%n without overflow
 */
int power_and_mod(int b, int e, int n) {
    long int currNum = 1;
    for (int i = 0; i < e; i++) {
        currNum *= b;
        if (currNum >= n) {
            currNum %= n;
        }
    }
    return (int) (currNum % n);
}
