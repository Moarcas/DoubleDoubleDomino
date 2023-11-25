
def main():
    path_true_labels = '../date/antrenare/'
    path_my_labels = '../date/351_Moarcas_Cosmin/'
    
    for joc in range(1, 6):
        for i in range(1, 21):
            if i < 10:
                file_name = f"{joc}_0{i}.txt"
            else:
                file_name = f"{joc}_{i}.txt"

            file_true_labels = open(path_true_labels + file_name, 'r')
            file_my_labels = open(path_my_labels + file_name, 'r')
            true_labels = file_true_labels.readlines()
            my_labels = file_my_labels.readlines()

            for k in range(2):
                my_elements = my_labels[k].split()
                true_elements = true_labels[k].split()

                true_position = true_elements[0]
                my_position = my_elements[0]
                true_nr_dots = true_elements[1]
                my_nr_dots = my_elements[1]

                if my_position != true_position:
                    print(file_name)
                    print('Pozitie gresita!')
                    return
                if my_nr_dots != true_nr_dots:
                    print(file_name)
                    print('Numar puncte gresit')
                    return
            
            true_points = true_labels[2].split()[0]
            my_points = my_labels[2].split()[0]
            if true_points != my_points:
                print(file_name)
                print('Punctaj gresit')
                return


if __name__ == '__main__':
    main()

