def array_manipulation(ar):
    ''' find the number that is divisible by 3
    args: array/list 
    return: list of number that is divisible by 3 
    '''
    result_list = [] 
    for a in ar: 
        if a%3 == 0:
            result_list.append(a)
    return result_list 
