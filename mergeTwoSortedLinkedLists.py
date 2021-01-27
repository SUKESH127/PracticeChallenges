# Merge Two Sorted Linked Lists
def mergeTwoSortedLists(l1, l2):
    dummyHead = cur = ListNode()
    while l1 != None and l2 != None:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if not l2 else l2
    return dummyHead.next
