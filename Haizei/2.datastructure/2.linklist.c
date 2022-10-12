#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef struct ListNode
{
    int data;
    struct ListNode *next;
} ListNode;

typedef struct LinkList
{
    ListNode head;
    int length;
} LinkList;

ListNode *init_listnode(int val) {
    ListNode *p = (ListNode *)malloc(sizeof(ListNode));
    p->data = val;
    p->next = NULL;
    return p;
}

LinkList *init_linklist() {
    LinkList *l = (LinkList *)malloc(sizeof(LinkList));
    l->head.next = NULL;
    l->length = 0;
    return l;
}

int insert(LinkList *l, int ind, int val) {
    if (l == NULL) return 0;
    if (ind > l->length || ind < 0) return 0;
    ListNode *p = &(l->head), *node = init_listnode(val);
    while(ind--) p = p->next;
    node->next = p->next;
    p->next = node;
    l->length += 1;
    return 1;
}

void clear_listnode(ListNode *node) {
    if (node == NULL) return ;
    free(node);
    return ;
}

void clear_linklist(LinkList *l) {
    if (l == NULL) return ;
    ListNode *p = l->head.next, *q;
    while (p)
    {
        q = p->next;
        clear_listnode(p);
        p = q;
    }
    free(l);
    return ;
}


int erase(LinkList *l, int ind) {
    if (l == NULL) return 0;
    if (ind >= l->length || ind < 0) return 0;
    ListNode *p = &(l->head), *q;
    while(ind--) p = p->next;
    q = p->next->next;
    clear_listnode(p->next);
    p->next = q;
    l->length -= 1;
    return 1;
}

void output(LinkList *l) {
	printf("linklist(%d)", l->length);
	for (ListNode *p = l->head.next; p; p = p->next) {
		printf("%d -> ", p->data); 
	}
	printf("NULL\n");
	return ;
}

int main() {
	srand(time(0));
	#define MAX_OP 20
	LinkList *l = init_linklist();
	int op, ind, val;
	for (int i = 0; i < MAX_OP; i++) {
		op = rand() % 3;
		ind = rand() % (l->length + 1);
		val = rand() % 100;
		switch (op) {
			case 0:
			case 1: {
				printf("insert %d at %d to linklist = %d\n", val, ind, insert(l, ind, val));
			} break;
			case 2: {
				printf("erase item at %d from linklist = %d\n", ind, erase(l, ind));
			} break;			
		}
		output(l);
        printf("\n");
	}
    clear_linklist(l);
    return 0;
}